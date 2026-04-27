"""
Structural analysis: for each answer part, analyse what the student answered
using a local LLM and a structured prompt.

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.structural_analysis \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

Inputs:  <output_folder>/answer_parts.jsonl
         <output_folder>/rubric_spec.json
Outputs: <output_folder>/answer_analysis.jsonl
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
    PipelineConfig,
    RubricSpec,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "structural_analysis_prompt.md"


def _load_answer_parts(path: Path) -> list[AnswerPart]:
    parts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                parts.append(AnswerPart.model_validate_json(line))
    return parts


def _load_rubric_spec(path: Path) -> RubricSpec:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RubricSpec.model_validate(data)


def _build_prompt(part: AnswerPart, rubric: RubricSpec, template: str) -> str:
    """Render the structural analysis prompt for one answer part."""
    qp_spec = rubric.get_part_spec(part.question_id, part.part_id)
    if qp_spec is None:
        task_instruction = f"Answer question {part.question_id}{part.part_id}"
        criteria_text = "(No rubric criteria available)"
    else:
        task_instruction = qp_spec.task_instruction
        criteria_lines = []
        for c in qp_spec.criteria:
            req = "; ".join(c.required_evidence) if c.required_evidence else "none specified"
            criteria_lines.append(
                f"- {c.name} ({c.max_marks} marks): {c.description}\n"
                f"  Required evidence: {req}\n"
                f"  Common misconceptions: {'; '.join(c.common_misconceptions) or 'none'}"
            )
        criteria_text = "\n".join(criteria_lines) if criteria_lines else "(No criteria)"

    if template:
        return (
            template
            .replace("{question_instruction}", task_instruction)
            .replace("{rubric_criteria}", criteria_text)
            .replace("{student_answer}", part.cleaned_text or part.raw_text)
        )

    # Fallback prompt if template file is missing
    return (
        f"You are a university marking assistant.\n\n"
        f"QUESTION INSTRUCTION:\n{task_instruction}\n\n"
        f"RUBRIC CRITERIA:\n{criteria_text}\n\n"
        f"STUDENT ANSWER:\n{part.cleaned_text or part.raw_text}\n\n"
        "Analyse the answer and return a JSON object with these keys:\n"
        "detected_claims, required_concepts_present, required_concepts_missing, "
        "reasoning_steps_present, reasoning_steps_missing, evidence_or_examples, "
        "calculation_or_method_steps, misconceptions, irrelevant_material, "
        "unsupported_assertions, quality_flags, model_confidence (0-1), needs_human_review (bool).\n"
        "Return ONLY the JSON."
    )


def _parse_analysis_response(
    raw: str,
    part: AnswerPart,
    client: OllamaClient,
    model: str,
    prompt: str,
    temperature: float,
) -> AnswerAnalysis:
    """Parse LLM JSON response into AnswerAnalysis, with one retry on failure."""
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text

    raw_json = _extract_json(raw)
    try:
        data = json.loads(raw_json)
        return AnswerAnalysis(
            submission_id=part.submission_id,
            question_id=part.question_id,
            part_id=part.part_id,
            **{k: v for k, v in data.items() if k not in ("submission_id", "question_id", "part_id")},
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning(
            "JSON parse failed for %s %s%s: %s — retrying",
            part.submission_id, part.question_id, part.part_id, exc,
        )

    # Retry with explicit JSON instruction
    retry_prompt = prompt + "\n\nYour response MUST be valid JSON only. No markdown, no explanation."
    raw2 = client.generate(model=model, prompt=retry_prompt, temperature=0.0)
    raw2_json = _extract_json(raw2)
    try:
        data = json.loads(raw2_json)
        return AnswerAnalysis(
            submission_id=part.submission_id,
            question_id=part.question_id,
            part_id=part.part_id,
            **{k: v for k, v in data.items() if k not in ("submission_id", "question_id", "part_id")},
        )
    except Exception as exc2:
        logger.error(
            "JSON parse failed after retry for %s %s%s: %s — flagging for human review",
            part.submission_id, part.question_id, part.part_id, exc2,
        )
        return AnswerAnalysis(
            submission_id=part.submission_id,
            question_id=part.question_id,
            part_id=part.part_id,
            quality_flags=["LLM analysis failed to parse"],
            model_confidence=0.0,
            needs_human_review=True,
        )


def run_structural_analysis(config: PipelineConfig, force: bool = False) -> Path:
    """
    Run structural analysis on all answer parts.

    Returns path to answer_analysis.jsonl.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "answer_analysis.jsonl"

    if output_path.exists() and not force:
        logger.info("answer_analysis.jsonl exists — skipping (use --force to overwrite)")
        return output_path

    answer_parts_path = output_dir / "answer_parts.jsonl"
    rubric_path = output_dir / "rubric_spec.json"

    if not answer_parts_path.exists():
        raise FileNotFoundError(f"answer_parts.jsonl not found at {answer_parts_path}. Run ingest first.")
    if not rubric_path.exists():
        raise FileNotFoundError(f"rubric_spec.json not found at {rubric_path}. Run rubric_compiler first.")

    parts = _load_answer_parts(answer_parts_path)
    rubric = _load_rubric_spec(rubric_path)

    template = ""
    if _PROMPT_PATH.exists():
        template = _PROMPT_PATH.read_text(encoding="utf-8")
    else:
        logger.warning("Structural analysis prompt template not found at %s — using fallback", _PROMPT_PATH)

    client = OllamaClient(
        base_url=config.ollama.base_url,
        timeout=config.ollama.timeout,
        stage="structural_analysis",
    )

    analyses: list[AnswerAnalysis] = []
    for i, part in enumerate(parts):
        logger.info("Analysing %d/%d: %s %s%s", i + 1, len(parts), part.submission_id, part.question_id, part.part_id)
        prompt = _build_prompt(part, rubric, template)
        raw = client.generate(
            model=config.models.analysis_model,
            prompt=prompt,
            temperature=config.temperature,
        )
        analysis = _parse_analysis_response(raw, part, client, config.models.analysis_model, prompt, config.temperature)
        analyses.append(analysis)

    with open(output_path, "w", encoding="utf-8") as fh:
        for a in analyses:
            fh.write(a.model_dump_json() + "\n")

    confidences = [a.model_confidence for a in analyses]
    if confidences:
        logger.info(
            "Analysis complete: %d answers | confidence mean=%.2f median=%.2f min=%.2f max=%.2f",
            len(analyses),
            sum(confidences) / len(confidences),
            statistics.median(confidences),
            min(confidences),
            max(confidences),
        )
    flagged = sum(1 for a in analyses if a.needs_human_review)
    if flagged:
        logger.warning("%d/%d answers flagged for human review", flagged, len(analyses))

    logger.info("Wrote %d analysis records to %s", len(analyses), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run structural analysis on answer parts")
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

    out = run_structural_analysis(cfg, force=args.force)
    print(f"Structural analysis complete: {out}")
    sys.exit(0)
