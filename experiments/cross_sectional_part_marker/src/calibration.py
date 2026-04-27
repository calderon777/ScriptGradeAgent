"""
Pairwise calibration of scores using cross-sectional structure.

For borderline and low-confidence answers, and NN pairs with score discrepancy > 1:
  - Run pairwise LLM comparison using calibration_prompt.md
  - Update calibrated_score only when both structural analysis and LLM agree
  - Document every change and every non-change

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.calibration \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerPart,
    CalibratedAnswer,
    CrossSectionalStructure,
    EmbeddingFeatures,
    PipelineConfig,
    ScoredAnswer,
)
from experiments.cross_sectional_part_marker.src.embeddings import answer_key

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "calibration_prompt.md"
_DISCREPANCY_THRESHOLD = 1.0  # marks


def _load_jsonl(path: Path, model_cls):
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(model_cls.model_validate_json(line))
    return records


def _load_cross_sectional(path: Path) -> list[CrossSectionalStructure]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [CrossSectionalStructure.model_validate(d) for d in data]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if (na > 0 and nb > 0) else 0.0


def _build_pairwise_prompt(
    question_instruction: str,
    rubric_criteria: str,
    answer_a_text: str,
    answer_a_score: float,
    answer_b_text: str,
    answer_b_score: float,
    template: str,
) -> str:
    if template:
        return (
            template
            .replace("{question_instruction}", question_instruction)
            .replace("{rubric_criteria}", rubric_criteria)
            .replace("{answer_a}", f"Score: {answer_a_score}\n\n{answer_a_text}")
            .replace("{answer_b}", f"Score: {answer_b_score}\n\n{answer_b_text}")
        )
    return (
        f"QUESTION: {question_instruction}\n\n"
        f"RUBRIC CRITERIA:\n{rubric_criteria}\n\n"
        f"ANSWER A (current score: {answer_a_score}):\n{answer_a_text}\n\n"
        f"ANSWER B (current score: {answer_b_score}):\n{answer_b_text}\n\n"
        "Compare these two answers against the rubric criteria.\n"
        "Return JSON: {\"recommendation\": \"keep_both|raise_A|lower_A|raise_B|lower_B\", "
        "\"reasoning\": str, \"confidence\": float (0-1)}"
    )


def _call_pairwise(
    client: OllamaClient,
    model: str,
    prompt: str,
    temperature: float,
) -> dict:
    raw = client.generate(model=model, prompt=prompt, temperature=temperature)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    logger.warning("Pairwise comparison returned invalid JSON")
    return {"recommendation": "keep_both", "reasoning": "parse_failed", "confidence": 0.0}


def run_calibration(config: PipelineConfig, force: bool = False) -> Path:
    """
    Calibrate scores using pairwise LLM comparisons.

    Returns path to calibrated_scores.jsonl.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "calibrated_scores.jsonl"

    if output_path.exists() and not force:
        logger.info("calibrated_scores.jsonl exists — skipping (use --force to overwrite)")
        return output_path

    scores_path = output_dir / "criterion_scores.jsonl"
    embedding_path = output_dir / "embedding_features.jsonl"
    structure_path = output_dir / "cross_sectional_structure.json"
    answer_parts_path = output_dir / "answer_parts.jsonl"
    rubric_path = output_dir / "rubric_spec.json"

    if not scores_path.exists():
        raise FileNotFoundError("criterion_scores.jsonl not found. Run scoring first.")

    scored: list[ScoredAnswer] = _load_jsonl(scores_path, ScoredAnswer)
    embedding_feats: list[EmbeddingFeatures] = _load_jsonl(embedding_path, EmbeddingFeatures)
    structures = _load_cross_sectional(structure_path)
    parts: list[AnswerPart] = _load_jsonl(answer_parts_path, AnswerPart)

    # Build lookup maps
    score_map: dict[str, ScoredAnswer] = {
        answer_key(s.submission_id, s.question_id, s.part_id): s for s in scored
    }
    feat_map: dict[str, EmbeddingFeatures] = {
        answer_key(f.submission_id, f.question_id, f.part_id): f for f in embedding_feats
    }
    part_map: dict[str, AnswerPart] = {
        answer_key(p.submission_id, p.question_id, p.part_id): p for p in parts
    }

    # Build borderline / low-confidence sets from structure
    borderline_set: set[str] = set()
    for struct in structures:
        for sid in struct.borderline_cases:
            borderline_set.add(answer_key(sid, struct.question_id, struct.part_id))

    rubric_spec = None
    if rubric_path.exists():
        with open(rubric_path, "r", encoding="utf-8") as fh:
            from experiments.cross_sectional_part_marker.src.schemas import RubricSpec
            rubric_spec = RubricSpec.model_validate(json.load(fh))

    template = _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""

    client: OllamaClient | None = None
    if config.run_pairwise_boundary:
        client = OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
            stage="calibration",
        )

    calibrated: list[CalibratedAnswer] = []

    for s in scored:
        key = answer_key(s.submission_id, s.question_id, s.part_id)
        feat = feat_map.get(key)
        part = part_map.get(key)
        is_borderline = key in borderline_set or s.needs_human_review
        is_low_conf = s.confidence < config.confidence_threshold

        pairwise_checks: list[dict] = []
        new_score = s.raw_total
        new_bucket = s.proposed_bucket
        calibration_notes_parts: list[str] = []
        nn_consistency = "not_checked"
        nn_discrepancies: list[tuple[str, float]] = []

        # Check NN pairs for discrepancies (always, regardless of pairwise LLM setting)
        if feat and feat.nearest_neighbours:
            for nn_id in feat.nearest_neighbours[:5]:
                nn_key = answer_key(nn_id, s.question_id, s.part_id)
                nn_scored = score_map.get(nn_key)
                if nn_scored and abs(nn_scored.raw_total - s.raw_total) > _DISCREPANCY_THRESHOLD:
                    nn_discrepancies.append((nn_id, nn_scored.raw_total))

            if not nn_discrepancies:
                nn_consistency = "consistent"

        # Flag discrepancies regardless of whether pairwise LLM is enabled
        if nn_discrepancies:
            nn_consistency = f"discrepancy_found ({len(nn_discrepancies)} pairs)"
            calibration_notes_parts.append(
                f"Score discrepancy found with {len(nn_discrepancies)} nearest neighbours"
            )

        # Run pairwise LLM comparison only if enabled and discrepancies found
        if nn_discrepancies and config.run_pairwise_boundary and client:
            # Run pairwise comparison for the most discrepant pair
            worst_nn_id, worst_nn_score = max(nn_discrepancies, key=lambda x: abs(x[1] - s.raw_total))
            nn_key = answer_key(worst_nn_id, s.question_id, s.part_id)
            nn_part = part_map.get(nn_key)
            this_part = part_map.get(key)

            if nn_part and this_part and rubric_spec:
                spec = rubric_spec.get_part_spec(s.question_id, s.part_id)
                if spec:
                    criteria_text = " ".join(c.name + ": " + c.description for c in spec.criteria)
                    prompt = _build_pairwise_prompt(
                        question_instruction=spec.task_instruction,
                        rubric_criteria=criteria_text,
                        answer_a_text=this_part.cleaned_text,
                        answer_a_score=s.raw_total,
                        answer_b_text=nn_part.cleaned_text,
                        answer_b_score=worst_nn_score,
                        template=template,
                    )
                    pairwise_result = _call_pairwise(client, config.models.scoring_model, prompt, config.temperature)
                    pairwise_checks.append({
                        "compared_to": worst_nn_id,
                        "their_score": worst_nn_score,
                        "recommendation": pairwise_result.get("recommendation"),
                        "reasoning": pairwise_result.get("reasoning"),
                        "confidence": pairwise_result.get("confidence"),
                    })

                    rec = pairwise_result.get("recommendation", "keep_both")
                    pw_conf = float(pairwise_result.get("confidence", 0.0))
                    if pw_conf >= 0.7:
                        if rec == "lower_A":
                            adjustment = min(0.5, (s.raw_total - worst_nn_score) / 2)
                            new_score = max(0, s.raw_total - adjustment)
                            calibration_notes_parts.append(f"Score lowered by {adjustment:.2f} based on pairwise comparison (confidence={pw_conf:.2f})")
                        elif rec == "raise_A":
                            adjustment = min(0.5, (worst_nn_score - s.raw_total) / 2)
                            new_score = min(s.max_total, s.raw_total + adjustment)
                            calibration_notes_parts.append(f"Score raised by {adjustment:.2f} based on pairwise comparison (confidence={pw_conf:.2f})")
                    else:
                        calibration_notes_parts.append(f"Pairwise recommendation '{rec}' with low confidence ({pw_conf:.2f}) — score unchanged")

        if not calibration_notes_parts:
            calibration_notes_parts.append("No calibration changes required")

        # Recompute bucket after calibration
        from experiments.cross_sectional_part_marker.src.scoring import score_to_bucket
        if rubric_spec:
            spec = rubric_spec.get_part_spec(s.question_id, s.part_id)
            if spec:
                new_bucket = score_to_bucket(new_score, spec.bucket_definitions)

        calibrated.append(
            CalibratedAnswer(
                submission_id=s.submission_id,
                anonymised_student_id=part.anonymised_student_id if part else "",
                question_id=s.question_id,
                part_id=s.part_id,
                initial_score=s.raw_total,
                calibrated_score=round(new_score, 2),
                max_score=s.max_total,
                initial_bucket=s.proposed_bucket,
                calibrated_bucket=new_bucket,
                calibration_notes="; ".join(calibration_notes_parts),
                nearest_neighbour_consistency=nn_consistency,
                pairwise_checks=pairwise_checks,
                confidence=s.confidence,
                needs_human_review=(is_borderline or is_low_conf or bool(nn_discrepancies if feat else False)),
            )
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        for c in calibrated:
            fh.write(c.model_dump_json() + "\n")

    changed = sum(1 for c in calibrated if abs(c.initial_score - c.calibrated_score) > 0.01)
    logger.info(
        "Calibration complete: %d answers | %d scores changed | %d flagged for review",
        len(calibrated), changed, sum(1 for c in calibrated if c.needs_human_review),
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate scores with pairwise LLM comparisons")
    p.add_argument("--config", required=True, help="Path to YAML pipeline config")
    p.add_argument("--force", action="store_true", help="Overwrite existing output")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    import sys

    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    cfg = PipelineConfig.model_validate(raw_cfg)

    out = run_calibration(cfg, force=args.force)
    print(f"Calibration complete: {out}")
    sys.exit(0)
