"""
Generate student feedback for each calibrated answer.

Tone: professional university marking — specific, grounded in evidence,
never overpraising, never inventing strengths not present in the answer.

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.feedback \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerAnalysis,
    AnswerPart,
    CalibratedAnswer,
    Feedback,
    FeedbackContent,
    PipelineConfig,
    RubricSpec,
)
from experiments.cross_sectional_part_marker.src.embeddings import answer_key

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "feedback_prompt.md"


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


def _build_feedback_prompt(
    part: AnswerPart,
    calibrated: CalibratedAnswer,
    analysis: AnswerAnalysis | None,
    rubric: RubricSpec,
    template: str,
) -> str:
    spec = rubric.get_part_spec(calibrated.question_id, calibrated.part_id)
    task_instruction = spec.task_instruction if spec else f"Question {calibrated.question_id}{calibrated.part_id}"
    criteria_text = (
        "\n".join(
            f"- {c.name} ({c.max_marks} marks): {c.description}"
            for c in spec.criteria
        )
        if spec else "(no criteria)"
    )

    analysis_summary = ""
    if analysis:
        analysis_summary = (
            f"Concepts present: {', '.join(analysis.required_concepts_present) or 'none'}\n"
            f"Concepts missing: {', '.join(analysis.required_concepts_missing) or 'none'}\n"
            f"Misconceptions: {', '.join(analysis.misconceptions) or 'none'}\n"
            f"Evidence cited: {', '.join(analysis.evidence_or_examples) or 'none'}"
        )

    student_answer = part.cleaned_text or part.raw_text
    score = calibrated.calibrated_score
    max_score = calibrated.initial_score  # preserve rubric max (initial is from scoring stage max_total)
    # Better: fetch from rubric spec
    if spec:
        max_score = spec.max_marks
    bucket = calibrated.calibrated_bucket

    if template:
        return (
            template
            .replace("{question_instruction}", task_instruction)
            .replace("{rubric_criteria}", criteria_text)
            .replace("{criterion_scores}", calibrated.calibration_notes)
            .replace("{structural_analysis}", analysis_summary)
            .replace("{student_answer}", student_answer[:2000])
            .replace("{score}", str(score))
            .replace("{max_score}", str(max_score))
            .replace("{bucket}", bucket)
        )

    return (
        f"You are a university marker writing feedback for a student.\n"
        f"Be specific, grounded, and professional. Do not overpraise.\n\n"
        f"QUESTION: {task_instruction}\n\n"
        f"CRITERIA:\n{criteria_text}\n\n"
        f"STRUCTURAL ANALYSIS:\n{analysis_summary}\n\n"
        f"STUDENT ANSWER:\n{student_answer[:2000]}\n\n"
        f"SCORE: {score}/{max_score} — Bucket: {bucket}\n\n"
        "Write feedback with these keys in a JSON object:\n"
        "- strengths: 1-2 specific sentences about what the student did well\n"
        "- limitations: 1-2 specific sentences about what was missing or weak\n"
        "- improvement_advice: 1-2 actionable sentences for improvement\n"
        "- summary: 1 sentence overall summary\n"
        "Return ONLY valid JSON. Do not invent strengths not evidenced in the answer."
    )


def _parse_feedback_response(
    raw: str,
    part: AnswerPart,
    calibrated: CalibratedAnswer,
    rubric: RubricSpec,
    client: OllamaClient,
    model: str,
    prompt: str,
    temperature: float,
) -> Feedback:
    """Parse LLM feedback response; retry once on failure."""
    spec = rubric.get_part_spec(calibrated.question_id, calibrated.part_id)
    max_score = spec.max_marks if spec else calibrated.initial_score

    def _extract(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end] if start >= 0 and end > start else text

    def _from_data(data: dict) -> Feedback:
        fb = FeedbackContent(
            strengths=data.get("strengths", ""),
            limitations=data.get("limitations", ""),
            improvement_advice=data.get("improvement_advice", ""),
            summary=data.get("summary", ""),
        )
        return Feedback(
            submission_id=part.submission_id,
            question_id=calibrated.question_id,
            part_id=calibrated.part_id,
            score=calibrated.calibrated_score,
            max_score=max_score,
            bucket=calibrated.calibrated_bucket,
            feedback=fb,
            evidence_used=[],
            model_confidence=calibrated.confidence,
            needs_human_review=calibrated.needs_human_review,
        )

    raw_json = _extract(raw)
    try:
        return _from_data(json.loads(raw_json))
    except Exception as exc:
        logger.warning("Feedback parse failed for %s: %s — retrying", part.submission_id, exc)

    retry_prompt = prompt + "\n\nReturn ONLY valid JSON. No markdown, no explanation."
    raw2 = client.generate(model=model, prompt=retry_prompt, temperature=0.0)
    raw2_json = _extract(raw2)
    try:
        return _from_data(json.loads(raw2_json))
    except Exception as exc2:
        logger.error("Feedback failed after retry for %s: %s", part.submission_id, exc2)
        return Feedback(
            submission_id=part.submission_id,
            question_id=calibrated.question_id,
            part_id=calibrated.part_id,
            score=calibrated.calibrated_score,
            max_score=max_score,
            bucket=calibrated.calibrated_bucket,
            feedback=FeedbackContent(
                strengths="",
                limitations="Feedback generation failed — needs manual review.",
                improvement_advice="",
                summary="Feedback generation failed.",
            ),
            model_confidence=0.0,
            needs_human_review=True,
        )


def run_feedback(config: PipelineConfig, force: bool = False) -> Path:
    """
    Generate feedback for all calibrated answers.

    Returns path to feedback.jsonl.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "feedback.jsonl"

    if output_path.exists() and not force:
        logger.info("feedback.jsonl exists — skipping (use --force to overwrite)")
        return output_path

    calibrated_path = output_dir / "calibrated_scores.jsonl"
    answer_parts_path = output_dir / "answer_parts.jsonl"
    analysis_path = output_dir / "answer_analysis.jsonl"
    rubric_path = output_dir / "rubric_spec.json"

    if not calibrated_path.exists():
        raise FileNotFoundError("calibrated_scores.jsonl not found. Run calibration first.")
    if not rubric_path.exists():
        raise FileNotFoundError("rubric_spec.json not found. Run rubric_compiler first.")

    calibrated_list: list[CalibratedAnswer] = _load_jsonl(calibrated_path, CalibratedAnswer)
    parts: list[AnswerPart] = _load_jsonl(answer_parts_path, AnswerPart)
    analyses: list[AnswerAnalysis] = _load_jsonl(analysis_path, AnswerAnalysis)

    with open(rubric_path, "r", encoding="utf-8") as fh:
        rubric = RubricSpec.model_validate(json.load(fh))

    part_map: dict[str, AnswerPart] = {
        answer_key(p.submission_id, p.question_id, p.part_id): p for p in parts
    }
    analysis_map: dict[str, AnswerAnalysis] = {
        answer_key(a.submission_id, a.question_id, a.part_id): a for a in analyses
    }

    template = _PROMPT_PATH.read_text(encoding="utf-8") if _PROMPT_PATH.exists() else ""

    client = OllamaClient(
        base_url=config.ollama.base_url,
        timeout=config.ollama.timeout,
        stage="feedback",
    )

    feedbacks: list[Feedback] = []
    for i, cal in enumerate(calibrated_list):
        key = answer_key(cal.submission_id, cal.question_id, cal.part_id)
        part = part_map.get(key)
        analysis = analysis_map.get(key)

        if part is None:
            logger.warning("No AnswerPart found for %s — creating placeholder", key)
            part = AnswerPart(
                submission_id=cal.submission_id,
                anonymised_student_id="unknown",
                source_file="",
                question_id=cal.question_id,
                part_id=cal.part_id,
                raw_text="",
                cleaned_text="",
                word_count=0,
                extraction_confidence=0.0,
            )

        logger.info("Generating feedback %d/%d: %s %s%s", i + 1, len(calibrated_list), cal.submission_id, cal.question_id, cal.part_id)
        prompt = _build_feedback_prompt(part, cal, analysis, rubric, template)
        raw = client.generate(model=config.models.feedback_model, prompt=prompt, temperature=config.temperature)
        fb = _parse_feedback_response(raw, part, cal, rubric, client, config.models.feedback_model, prompt, config.temperature)
        feedbacks.append(fb)

    with open(output_path, "w", encoding="utf-8") as fh:
        for fb in feedbacks:
            fh.write(fb.model_dump_json() + "\n")

    flagged = sum(1 for fb in feedbacks if fb.needs_human_review)
    logger.info("Wrote %d feedback records to %s (%d flagged for review)", len(feedbacks), output_path, flagged)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate student feedback from calibrated scores")
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

    out = run_feedback(cfg, force=args.force)
    print(f"Feedback complete: {out}")
    sys.exit(0)
