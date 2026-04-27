"""
Score each answer part against rubric criteria using a local LLM.

Multi-step prompt strategy:
  1. Evidence extraction per criterion
  2. Score each criterion with justification
  3. Sum total and assign bucket
  4. Assign confidence
  5. Return structured JSON

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.scoring \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerAnalysis,
    AnswerPart,
    BucketDefinition,
    CriterionScore,
    EmbeddingFeatures,
    PipelineConfig,
    QuestionPartSpec,
    RubricSpec,
    ScoredAnswer,
)
from experiments.cross_sectional_part_marker.src.embeddings import answer_key

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "scoring_prompt.md"


def score_to_bucket(score: float, bucket_defs: list[BucketDefinition]) -> str:
    """
    Map a numeric score to the appropriate bucket label using bucket_definitions.

    Parameters
    ----------
    score:       The numeric score to map
    bucket_defs: List of BucketDefinition for the question part

    Returns
    -------
    Bucket letter (A–F), defaulting to "F" if no matching bucket found.
    """
    if not bucket_defs:
        return "F"
    # Sort by mark_range lower bound descending so highest bucket checked first
    sorted_defs = sorted(bucket_defs, key=lambda b: b.mark_range[0], reverse=True)
    for bdef in sorted_defs:
        lo, hi = bdef.mark_range
        if lo <= score <= hi:
            return bdef.bucket
    # Clamp to lowest bucket for scores below all ranges
    lowest = sorted(bucket_defs, key=lambda b: b.mark_range[0])
    return lowest[0].bucket if lowest else "F"


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


def _load_rubric(path: Path) -> RubricSpec:
    with open(path, "r", encoding="utf-8") as fh:
        return RubricSpec.model_validate(json.load(fh))


def _format_criteria(spec: QuestionPartSpec) -> str:
    lines = []
    for c in spec.criteria:
        lines.append(
            f"Criterion: {c.name} (ID: {c.criterion_id}, max: {c.max_marks} marks)\n"
            f"  Description: {c.description}\n"
            f"  Required evidence: {'; '.join(c.required_evidence) or 'none'}\n"
            f"  Partial credit: {'; '.join(c.partial_credit_rules) or 'none'}\n"
            f"  No-credit conditions: {'; '.join(c.no_credit_conditions) or 'none'}"
        )
    return "\n\n".join(lines)


def _format_bucket_defs(spec: QuestionPartSpec) -> str:
    return "\n".join(
        f"  {b.bucket} ({b.label}): {b.mark_range[0]}–{b.mark_range[1]} marks — {b.description}"
        for b in spec.bucket_definitions
    )


def _build_scoring_prompt(
    part: AnswerPart,
    spec: QuestionPartSpec,
    analysis: AnswerAnalysis | None,
    template: str,
) -> str:
    criteria_text = _format_criteria(spec)
    bucket_text = _format_bucket_defs(spec)
    analysis_text = ""
    if analysis:
        analysis_text = (
            f"Concepts present: {', '.join(analysis.required_concepts_present)}\n"
            f"Concepts missing: {', '.join(analysis.required_concepts_missing)}\n"
            f"Misconceptions: {', '.join(analysis.misconceptions)}\n"
            f"Quality flags: {', '.join(analysis.quality_flags)}"
        )

    if template:
        return (
            template
            .replace("{question_instruction}", spec.task_instruction)
            .replace("{rubric_criteria}", criteria_text)
            .replace("{criterion_definitions}", criteria_text)
            .replace("{structural_analysis}", analysis_text)
            .replace("{student_answer}", part.cleaned_text or part.raw_text)
            .replace("{bucket_definitions}", bucket_text)
        )

    return (
        f"You are a university marker. Score the student answer against each criterion.\n\n"
        f"QUESTION: {spec.task_instruction}\n\n"
        f"CRITERIA:\n{criteria_text}\n\n"
        f"BUCKET DEFINITIONS:\n{bucket_text}\n\n"
        f"STRUCTURAL ANALYSIS:\n{analysis_text}\n\n"
        f"STUDENT ANSWER:\n{part.cleaned_text or part.raw_text}\n\n"
        "Return a JSON object:\n"
        '{"criterion_scores": [{"criterion_id": str, "score": float, "max_score": float, '
        '"evidence": str, "reason": str, "confidence": float}], '
        '"raw_total": float, "max_total": float, "proposed_bucket": str, '
        '"confidence": float, "needs_human_review": bool, "review_reason": str}\n'
        "Scores must not exceed max_score per criterion. Return ONLY JSON."
    )


def _parse_scoring_response(
    raw: str,
    part: AnswerPart,
    spec: QuestionPartSpec,
    client: OllamaClient,
    model: str,
    prompt: str,
    temperature: float,
    config: PipelineConfig,
) -> ScoredAnswer:
    """Parse LLM response into ScoredAnswer; enforce constraints; retry once on failure."""

    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end] if start >= 0 and end > start else text

    def _build_from_data(data: dict) -> ScoredAnswer:
        raw_criterion_scores = data.get("criterion_scores", [])
        criterion_scores: list[CriterionScore] = []
        for cs in raw_criterion_scores:
            cid = cs.get("criterion_id", "unknown")
            # Look up max score from rubric
            rubric_max = next((c.max_marks for c in spec.criteria if c.criterion_id == cid), None)
            max_score = float(cs.get("max_score", rubric_max or 0))
            score = min(float(cs.get("score", 0)), max_score)
            criterion_scores.append(
                CriterionScore(
                    criterion_id=cid,
                    score=score,
                    max_score=max_score,
                    evidence=cs.get("evidence", ""),
                    reason=cs.get("reason", ""),
                    confidence=float(cs.get("confidence", 0.5)),
                )
            )

        max_total = float(data.get("max_total", spec.max_marks))
        raw_total = min(float(data.get("raw_total", sum(cs.score for cs in criterion_scores))), max_total)
        proposed_bucket = data.get("proposed_bucket") or score_to_bucket(raw_total, spec.bucket_definitions)
        confidence = float(data.get("confidence", 0.5))
        needs_review = confidence < config.confidence_threshold or bool(data.get("needs_human_review", False))
        review_reason = data.get("review_reason", "")
        if confidence < config.confidence_threshold and not review_reason:
            review_reason = f"Low model confidence: {confidence:.2f}"

        return ScoredAnswer(
            submission_id=part.submission_id,
            question_id=part.question_id,
            part_id=part.part_id,
            criterion_scores=criterion_scores,
            raw_total=raw_total,
            max_total=max_total,
            proposed_bucket=proposed_bucket,
            confidence=confidence,
            needs_human_review=needs_review,
            review_reason=review_reason,
        )

    raw_json = _extract_json(raw)
    try:
        data = json.loads(raw_json)
        return _build_from_data(data)
    except Exception as exc:
        logger.warning("Scoring JSON parse failed for %s %s%s: %s — retrying", part.submission_id, part.question_id, part.part_id, exc)

    retry_prompt = prompt + "\n\nRespond ONLY with valid JSON. No explanation, no markdown."
    raw2 = client.generate(model=model, prompt=retry_prompt, temperature=0.0)
    raw2_json = _extract_json(raw2)
    try:
        data = json.loads(raw2_json)
        return _build_from_data(data)
    except Exception as exc2:
        logger.error("Scoring failed after retry for %s: %s", part.submission_id, exc2)
        return ScoredAnswer(
            submission_id=part.submission_id,
            question_id=part.question_id,
            part_id=part.part_id,
            raw_total=0.0,
            max_total=spec.max_marks,
            proposed_bucket="F",
            confidence=0.0,
            needs_human_review=True,
            review_reason="LLM scoring failed to parse after retry",
        )


def run_scoring(config: PipelineConfig, force: bool = False) -> Path:
    """
    Score all answer parts against rubric criteria.

    Returns path to criterion_scores.jsonl.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "criterion_scores.jsonl"

    if output_path.exists() and not force:
        logger.info("criterion_scores.jsonl exists — skipping (use --force to overwrite)")
        return output_path

    answer_parts_path = output_dir / "answer_parts.jsonl"
    rubric_path = output_dir / "rubric_spec.json"
    analysis_path = output_dir / "answer_analysis.jsonl"

    if not answer_parts_path.exists():
        raise FileNotFoundError("answer_parts.jsonl not found. Run ingest first.")
    if not rubric_path.exists():
        raise FileNotFoundError("rubric_spec.json not found. Run rubric_compiler first.")

    parts: list[AnswerPart] = _load_jsonl(answer_parts_path, AnswerPart)
    rubric = _load_rubric(rubric_path)
    analyses: list[AnswerAnalysis] = _load_jsonl(analysis_path, AnswerAnalysis) if analysis_path.exists() else []
    analysis_map: dict[str, AnswerAnalysis] = {
        answer_key(a.submission_id, a.question_id, a.part_id): a for a in analyses
    }

    template = _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""

    client = OllamaClient(
        base_url=config.ollama.base_url,
        timeout=config.ollama.timeout,
        stage="scoring",
    )

    scored: list[ScoredAnswer] = []
    for i, part in enumerate(parts):
        spec = rubric.get_part_spec(part.question_id, part.part_id)
        if spec is None:
            logger.warning("No rubric spec for %s %s%s — skipping", part.submission_id, part.question_id, part.part_id)
            continue

        analysis = analysis_map.get(answer_key(part.submission_id, part.question_id, part.part_id))
        logger.info("Scoring %d/%d: %s %s%s", i + 1, len(parts), part.submission_id, part.question_id, part.part_id)
        prompt = _build_scoring_prompt(part, spec, analysis, template)
        raw = client.generate(model=config.models.scoring_model, prompt=prompt, temperature=config.temperature)
        result = _parse_scoring_response(raw, part, spec, client, config.models.scoring_model, prompt, config.temperature, config)
        scored.append(result)

    with open(output_path, "w", encoding="utf-8") as fh:
        for s in scored:
            fh.write(s.model_dump_json() + "\n")

    confidences = [s.confidence for s in scored]
    if confidences:
        logger.info(
            "Scoring complete: %d answers | mean confidence=%.2f | median=%.2f | flagged=%d",
            len(scored),
            sum(confidences) / len(confidences),
            statistics.median(confidences),
            sum(1 for s in scored if s.needs_human_review),
        )
    logger.info("Wrote %d scored answers to %s", len(scored), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score answer parts against rubric criteria")
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

    out = run_scoring(cfg, force=args.force)
    print(f"Scoring complete: {out}")
    sys.exit(0)
