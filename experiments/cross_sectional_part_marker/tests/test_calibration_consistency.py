"""
Tests for calibration consistency logic.

These tests verify:
1. Calibration does NOT change scores when NN answers are consistent
2. Calibration DOES flag for review when similar answers have scores > 1 mark apart
3. Calibration notes are non-empty when a score is changed

Run with:
    pytest experiments/cross_sectional_part_marker/tests/test_calibration_consistency.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.cross_sectional_part_marker.src.calibration import (
    _DISCREPANCY_THRESHOLD,
    _build_pairwise_prompt,
    _call_pairwise,
)
from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerPart,
    BucketDefinition,
    CalibratedAnswer,
    CriterionScore,
    EmbeddingFeatures,
    PipelineConfig,
    QuestionPartSpec,
    RubricSpec,
    ScoredAnswer,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def _make_scored(
    submission_id: str,
    score: float,
    max_total: float = 5.0,
    confidence: float = 0.8,
    needs_review: bool = False,
) -> ScoredAnswer:
    return ScoredAnswer(
        submission_id=submission_id,
        question_id="Q1",
        part_id="a",
        raw_total=score,
        max_total=max_total,
        proposed_bucket="C",
        confidence=confidence,
        needs_human_review=needs_review,
        review_reason="",
    )


def _make_features(
    submission_id: str,
    neighbours: list[str],
    cluster_id: int = 0,
    outlier_score: float = 0.0,
) -> EmbeddingFeatures:
    return EmbeddingFeatures(
        submission_id=submission_id,
        question_id="Q1",
        part_id="a",
        embedding_model="nomic-embed-text",
        nearest_neighbours=neighbours,
        cluster_id=cluster_id,
        outlier_score=outlier_score,
    )


def _make_part(submission_id: str, text: str = "Some answer text.") -> AnswerPart:
    return AnswerPart(
        submission_id=submission_id,
        anonymised_student_id=f"ANON_{submission_id}",
        source_file=f"data/{submission_id}.txt",
        question_id="Q1",
        part_id="a",
        raw_text=text,
        cleaned_text=text,
        word_count=len(text.split()),
        extraction_confidence=0.9,
    )


def _make_rubric() -> RubricSpec:
    spec = QuestionPartSpec(
        question_id="Q1",
        part_id="a",
        max_marks=5.0,
        task_instruction="Explain demand.",
        criteria=[],
        bucket_definitions=[
            BucketDefinition(bucket="A", label="Excellent",  mark_range=(5, 5), description=""),
            BucketDefinition(bucket="B", label="Good",       mark_range=(4, 4), description=""),
            BucketDefinition(bucket="C", label="Adequate",   mark_range=(3, 3), description=""),
            BucketDefinition(bucket="D", label="Poor",       mark_range=(2, 2), description=""),
            BucketDefinition(bucket="E", label="Very Poor",  mark_range=(1, 1), description=""),
            BucketDefinition(bucket="F", label="Fail",       mark_range=(0, 0), description=""),
        ],
    )
    return RubricSpec(assessment_name="Test", question_parts=[spec])


# ---------------------------------------------------------------------------
# Test 1: No change when NNs are consistent
# ---------------------------------------------------------------------------


class TestConsistentNeighboursNoChange:
    """When all nearest neighbours have the same score, calibration should not change the score."""

    def test_consistent_nns_no_score_change(self, tmp_path):
        """
        Three answers with the same score (3.0) and each listed as NN of the others.
        Calibration should not change any score.
        """
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        subs = ["sub_A", "sub_B", "sub_C"]
        scores = {s: 3.0 for s in subs}

        # Write criterion_scores.jsonl
        scored_list = [_make_scored(s, 3.0, confidence=0.85) for s in subs]
        with open(output_dir / "criterion_scores.jsonl", "w", encoding="utf-8") as fh:
            for sc in scored_list:
                fh.write(sc.model_dump_json() + "\n")

        # Write embedding_features.jsonl (each points to others as NNs)
        features = [
            _make_features(subs[0], [subs[1], subs[2]]),
            _make_features(subs[1], [subs[0], subs[2]]),
            _make_features(subs[2], [subs[0], subs[1]]),
        ]
        with open(output_dir / "embedding_features.jsonl", "w", encoding="utf-8") as fh:
            for f in features:
                fh.write(f.model_dump_json() + "\n")

        # Write rubric_spec.json
        rubric = _make_rubric()
        with open(output_dir / "rubric_spec.json", "w", encoding="utf-8") as fh:
            fh.write(rubric.model_dump_json())

        # Write empty cross_sectional_structure
        with open(output_dir / "cross_sectional_structure.json", "w", encoding="utf-8") as fh:
            json.dump([], fh)

        # Write answer_parts.jsonl
        with open(output_dir / "answer_parts.jsonl", "w", encoding="utf-8") as fh:
            for s in subs:
                fh.write(_make_part(s).model_dump_json() + "\n")

        # Build config pointing to tmp_path outputs
        cfg_dict = {
            "submissions_folder": str(tmp_path / "subs"),
            "output_folder": str(output_dir),
            "marking_scheme_path": str(tmp_path / "scheme.txt"),
            "run_pairwise_boundary": False,  # Disable LLM pairwise to avoid needing Ollama
            "confidence_threshold": 0.65,
        }
        cfg = PipelineConfig.model_validate(cfg_dict)

        from experiments.cross_sectional_part_marker.src.calibration import run_calibration

        output_path = run_calibration(cfg, force=True)
        assert output_path.exists()

        # Load results
        results: list[CalibratedAnswer] = []
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(CalibratedAnswer.model_validate_json(line))

        assert len(results) == 3
        for r in results:
            assert abs(r.calibrated_score - 3.0) < 0.01, (
                f"{r.submission_id}: expected score 3.0, got {r.calibrated_score}"
            )
            assert r.nearest_neighbour_consistency in ("consistent", "not_checked"), (
                f"{r.submission_id}: unexpected nn_consistency: {r.nearest_neighbour_consistency}"
            )


# ---------------------------------------------------------------------------
# Test 2: Flag for review when similar answers have score discrepancy > 1
# ---------------------------------------------------------------------------


class TestDiscrepantNeighboursReview:
    """When two NNs have score discrepancy > 1 mark, the answer should be flagged for review."""

    def test_discrepancy_triggers_review_flag(self, tmp_path):
        """
        sub_high scores 4.5, sub_low scores 2.0 (discrepancy = 2.5 > 1.0).
        sub_high lists sub_low as a NN. Should be flagged for human review.
        """
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Write scores
        scored_list = [
            _make_scored("sub_high", 4.5, confidence=0.82),
            _make_scored("sub_low",  2.0, confidence=0.80),
        ]
        with open(output_dir / "criterion_scores.jsonl", "w", encoding="utf-8") as fh:
            for sc in scored_list:
                fh.write(sc.model_dump_json() + "\n")

        # sub_high has sub_low as nearest neighbour
        features = [
            _make_features("sub_high", ["sub_low"]),
            _make_features("sub_low",  ["sub_high"]),
        ]
        with open(output_dir / "embedding_features.jsonl", "w", encoding="utf-8") as fh:
            for f in features:
                fh.write(f.model_dump_json() + "\n")

        rubric = _make_rubric()
        with open(output_dir / "rubric_spec.json", "w", encoding="utf-8") as fh:
            fh.write(rubric.model_dump_json())

        with open(output_dir / "cross_sectional_structure.json", "w", encoding="utf-8") as fh:
            json.dump([], fh)

        with open(output_dir / "answer_parts.jsonl", "w", encoding="utf-8") as fh:
            fh.write(_make_part("sub_high", "Good answer demonstrating supply and demand correctly.").model_dump_json() + "\n")
            fh.write(_make_part("sub_low",  "Vague answer about economics.").model_dump_json() + "\n")

        cfg_dict = {
            "submissions_folder": str(tmp_path / "subs"),
            "output_folder": str(output_dir),
            "marking_scheme_path": str(tmp_path / "scheme.txt"),
            "run_pairwise_boundary": False,  # No LLM calls needed
            "confidence_threshold": 0.65,
        }
        cfg = PipelineConfig.model_validate(cfg_dict)

        from experiments.cross_sectional_part_marker.src.calibration import run_calibration

        output_path = run_calibration(cfg, force=True)
        results: list[CalibratedAnswer] = []
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(CalibratedAnswer.model_validate_json(line))

        sub_high_result = next(r for r in results if r.submission_id == "sub_high")

        # The discrepancy (4.5 - 2.0 = 2.5) > 1.0 should trigger review flag
        assert sub_high_result.needs_human_review is True, (
            "Expected sub_high to be flagged for human review due to NN score discrepancy"
        )
        assert "discrepancy" in sub_high_result.nearest_neighbour_consistency.lower(), (
            f"Expected 'discrepancy' in nn_consistency, got: {sub_high_result.nearest_neighbour_consistency}"
        )


# ---------------------------------------------------------------------------
# Test 3: Notes non-empty when score is changed
# ---------------------------------------------------------------------------


class TestCalibrationNotesOnChange:
    """Calibration notes must be non-empty whenever a score is changed."""

    def test_notes_non_empty_after_change(self, tmp_path):
        """
        Simulate a pairwise check that recommends lowering sub_high's score.
        After calibration, notes must be non-empty.
        """
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        scored_list = [
            _make_scored("sub_high", 4.0, confidence=0.75),
            _make_scored("sub_low",  2.0, confidence=0.75),
        ]
        with open(output_dir / "criterion_scores.jsonl", "w", encoding="utf-8") as fh:
            for sc in scored_list:
                fh.write(sc.model_dump_json() + "\n")

        features = [
            _make_features("sub_high", ["sub_low"]),
            _make_features("sub_low",  ["sub_high"]),
        ]
        with open(output_dir / "embedding_features.jsonl", "w", encoding="utf-8") as fh:
            for f in features:
                fh.write(f.model_dump_json() + "\n")

        rubric = _make_rubric()
        with open(output_dir / "rubric_spec.json", "w", encoding="utf-8") as fh:
            fh.write(rubric.model_dump_json())

        with open(output_dir / "cross_sectional_structure.json", "w", encoding="utf-8") as fh:
            json.dump([], fh)

        with open(output_dir / "answer_parts.jsonl", "w", encoding="utf-8") as fh:
            fh.write(_make_part("sub_high").model_dump_json() + "\n")
            fh.write(_make_part("sub_low").model_dump_json() + "\n")

        # Mock the OllamaClient.generate to return a "lower_A" recommendation with high confidence
        mock_response = json.dumps({
            "recommendation": "lower_A",
            "reasoning": "Answer B is actually of equivalent quality; sub_high was overscored.",
            "confidence": 0.85,
        })

        cfg_dict = {
            "submissions_folder": str(tmp_path / "subs"),
            "output_folder": str(output_dir),
            "marking_scheme_path": str(tmp_path / "scheme.txt"),
            "run_pairwise_boundary": True,
            "confidence_threshold": 0.65,
        }
        cfg = PipelineConfig.model_validate(cfg_dict)

        with patch(
            "experiments.cross_sectional_part_marker.src.calibration.OllamaClient"
        ) as MockClient:
            mock_instance = MagicMock()
            mock_instance.generate.return_value = mock_response
            MockClient.return_value = mock_instance

            from experiments.cross_sectional_part_marker.src.calibration import run_calibration

            output_path = run_calibration(cfg, force=True)

        results: list[CalibratedAnswer] = []
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(CalibratedAnswer.model_validate_json(line))

        sub_high_result = next(r for r in results if r.submission_id == "sub_high")

        # Score should have been lowered
        assert sub_high_result.calibrated_score < sub_high_result.initial_score, (
            "Expected score to be lowered for sub_high after pairwise comparison"
        )

        # Calibration notes must be non-empty
        assert sub_high_result.calibration_notes.strip() != "", (
            "Expected non-empty calibration notes when score was changed"
        )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestPairwiseHelpers:
    def test_build_pairwise_prompt_uses_template_placeholders(self):
        template = (
            "Q: {question_instruction}\n"
            "CRITERIA: {rubric_criteria}\n"
            "A: {answer_a}\n"
            "B: {answer_b}"
        )
        prompt = _build_pairwise_prompt(
            question_instruction="Explain demand.",
            rubric_criteria="Criterion 1: shift",
            answer_a_text="Demand shifts right.",
            answer_a_score=3.0,
            answer_b_text="Price goes up.",
            answer_b_score=1.5,
            template=template,
        )
        assert "Explain demand." in prompt
        assert "Criterion 1: shift" in prompt
        assert "3.0" in prompt
        assert "1.5" in prompt

    def test_discrepancy_threshold_value(self):
        """Verify the threshold constant is set to 1.0."""
        assert _DISCREPANCY_THRESHOLD == 1.0
