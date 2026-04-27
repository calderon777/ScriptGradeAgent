"""
Tests for schemas.py — validate Pydantic models work correctly.

Run with:
    pytest experiments/cross_sectional_part_marker/tests/test_schemas.py -v
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerPart,
    BucketDefinition,
    CriterionScore,
    HumanReviewEntry,
    QuestionPartSpec,
    RubricSpec,
    ScoredAnswer,
)

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# AnswerPart
# ---------------------------------------------------------------------------


class TestAnswerPart:
    def test_valid_answer_part(self):
        part = AnswerPart(
            submission_id="sub_001",
            anonymised_student_id="ANON_001",
            source_file="data/submissions/sub_001.pdf",
            question_id="Q1",
            part_id="a",
            raw_text="Supply and demand interact at equilibrium.",
            cleaned_text="Supply and demand interact at equilibrium.",
            word_count=6,
            extraction_confidence=0.9,
            extraction_notes="",
        )
        assert part.submission_id == "sub_001"
        assert part.word_count == 6
        assert part.extraction_confidence == 0.9

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            AnswerPart(
                submission_id="sub_001",
                # missing anonymised_student_id and others
            )

    def test_extraction_confidence_bounds(self):
        with pytest.raises(Exception):
            AnswerPart(
                submission_id="sub_001",
                anonymised_student_id="ANON_001",
                source_file="test.pdf",
                question_id="Q1",
                part_id="a",
                raw_text="text",
                cleaned_text="text",
                word_count=1,
                extraction_confidence=1.5,  # invalid: > 1.0
            )

    def test_extraction_confidence_negative_raises(self):
        with pytest.raises(Exception):
            AnswerPart(
                submission_id="sub_001",
                anonymised_student_id="ANON_001",
                source_file="test.pdf",
                question_id="Q1",
                part_id="a",
                raw_text="text",
                cleaned_text="text",
                word_count=1,
                extraction_confidence=-0.1,  # invalid
            )

    def test_serialisation_roundtrip(self):
        part = AnswerPart(
            submission_id="sub_999",
            anonymised_student_id="ANON_999",
            source_file="data/test.txt",
            question_id="Q2",
            part_id="b",
            raw_text="The market clears when supply equals demand.",
            cleaned_text="The market clears when supply equals demand.",
            word_count=8,
            extraction_confidence=0.75,
        )
        json_str = part.model_dump_json()
        reloaded = AnswerPart.model_validate_json(json_str)
        assert reloaded.submission_id == part.submission_id
        assert reloaded.extraction_confidence == part.extraction_confidence


# ---------------------------------------------------------------------------
# RubricSpec round-trip
# ---------------------------------------------------------------------------


class TestRubricSpec:
    def test_round_trip_from_fixture(self):
        fixture_path = _FIXTURES / "sample_rubric_spec.json"
        with open(fixture_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        spec = RubricSpec.model_validate(raw)

        # Should have 4 question parts
        assert len(spec.question_parts) == 4
        assert spec.assessment_name == "Introduction to Economics — Semester 1 Exam"
        assert spec.needs_human_confirmation is False

        # Re-serialise and re-parse
        json_str = spec.model_dump_json()
        spec2 = RubricSpec.model_validate_json(json_str)
        assert len(spec2.question_parts) == len(spec.question_parts)
        assert spec2.assessment_name == spec.assessment_name

    def test_get_part_spec(self):
        fixture_path = _FIXTURES / "sample_rubric_spec.json"
        with open(fixture_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        spec = RubricSpec.model_validate(raw)

        q1a = spec.get_part_spec("Q1", "a")
        assert q1a is not None
        assert q1a.max_marks == 5
        assert len(q1a.criteria) == 3

        q2b = spec.get_part_spec("Q2", "b")
        assert q2b is not None
        assert q2b.max_marks == 10

        missing = spec.get_part_spec("Q99", "z")
        assert missing is None

    def test_empty_rubric_spec(self):
        spec = RubricSpec(assessment_name="Test", question_parts=[])
        assert spec.needs_human_confirmation is False
        assert spec.question_parts == []


# ---------------------------------------------------------------------------
# ScoredAnswer enforces score <= max_score
# ---------------------------------------------------------------------------


class TestScoredAnswer:
    def test_valid_scored_answer(self):
        cs = CriterionScore(
            criterion_id="c1",
            score=3.0,
            max_score=4.0,
            evidence="Student mentions X",
            reason="Partial credit",
            confidence=0.8,
        )
        sa = ScoredAnswer(
            submission_id="sub_001",
            question_id="Q1",
            part_id="a",
            criterion_scores=[cs],
            raw_total=3.0,
            max_total=5.0,
            proposed_bucket="C",
            confidence=0.8,
        )
        assert sa.raw_total == 3.0
        assert sa.proposed_bucket == "C"

    def test_criterion_score_exceeds_max_raises(self):
        with pytest.raises(Exception):
            CriterionScore(
                criterion_id="c1",
                score=5.0,  # exceeds max_score
                max_score=4.0,
                evidence="",
                reason="",
                confidence=0.5,
            )

    def test_scored_answer_total_exceeds_max_raises(self):
        cs = CriterionScore(criterion_id="c1", score=2.0, max_score=5.0)
        with pytest.raises(Exception):
            ScoredAnswer(
                submission_id="sub_001",
                question_id="Q1",
                part_id="a",
                criterion_scores=[cs],
                raw_total=8.0,  # exceeds max_total of 5
                max_total=5.0,
                proposed_bucket="A",
                confidence=0.9,
            )

    def test_zero_raw_total_valid(self):
        sa = ScoredAnswer(
            submission_id="sub_004",
            question_id="Q1",
            part_id="a",
            raw_total=0.0,
            max_total=5.0,
            proposed_bucket="F",
        )
        assert sa.proposed_bucket == "F"
        assert sa.raw_total == 0.0


# ---------------------------------------------------------------------------
# HumanReviewEntry UUID generation
# ---------------------------------------------------------------------------


class TestHumanReviewEntry:
    def test_auto_uuid_generated(self):
        entry = HumanReviewEntry(
            timestamp="2025-01-01T12:00:00Z",
            reviewer="marker1",
            submission_id="sub_001",
            question_id="Q1",
            part_id="a",
            ai_score=3.5,
            ai_bucket="C",
        )
        assert entry.review_id is not None
        # Should be a valid UUID
        parsed = uuid.UUID(entry.review_id)
        assert str(parsed) == entry.review_id

    def test_explicit_uuid_preserved(self):
        custom_id = str(uuid.uuid4())
        entry = HumanReviewEntry(
            review_id=custom_id,
            timestamp="2025-01-01T12:00:00Z",
            reviewer="marker1",
            submission_id="sub_001",
            question_id="Q1",
            part_id="a",
            ai_score=3.5,
            ai_bucket="C",
        )
        assert entry.review_id == custom_id

    def test_two_entries_have_different_uuids(self):
        kwargs = dict(
            timestamp="2025-01-01T12:00:00Z",
            reviewer="marker1",
            submission_id="sub_001",
            question_id="Q1",
            part_id="a",
            ai_score=3.5,
            ai_bucket="C",
        )
        e1 = HumanReviewEntry(**kwargs)
        e2 = HumanReviewEntry(**kwargs)
        assert e1.review_id != e2.review_id

    def test_optional_human_fields_default_none(self):
        entry = HumanReviewEntry(
            timestamp="2025-01-01T12:00:00Z",
            reviewer="marker1",
            submission_id="sub_001",
            question_id="Q1",
            part_id="a",
            ai_score=3.5,
            ai_bucket="B",
        )
        assert entry.human_score is None
        assert entry.human_bucket is None
        assert entry.human_feedback is None
