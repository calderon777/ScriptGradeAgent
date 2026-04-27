"""
Compile a rubric specification from marking scheme / rubric / instructions files.

Accepts:
  - Structured JSON/YAML input  -> parsed directly
  - Free text (markdown, plain)  -> LLM extraction

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.rubric_compiler \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

Outputs:
    <output_folder>/rubric_spec.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
from experiments.cross_sectional_part_marker.src.schemas import (
    BucketDefinition,
    Criterion,
    PipelineConfig,
    QuestionPartSpec,
    RubricSpec,
)

logger = logging.getLogger(__name__)

_RUBRIC_COMPILER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "rubric_compiler_prompt.md"
)


def _load_file_text(path: Path) -> str:
    """Read a file, using PDF/DOCX extraction where needed."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from experiments.cross_sectional_part_marker.src.ingest import extract_text_from_pdf
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        from experiments.cross_sectional_part_marker.src.ingest import extract_text_from_docx
        return extract_text_from_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _try_parse_structured(text: str) -> dict | None:
    """Try JSON then YAML parsing; return dict or None."""
    text_stripped = text.strip()
    # Try JSON
    if text_stripped.startswith("{") or text_stripped.startswith("["):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass
    # Try YAML
    try:
        data = yaml.safe_load(text_stripped)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass
    return None


def _rubric_from_structured(data: dict, source: str) -> RubricSpec:
    """Build a RubricSpec from an already-structured dict."""
    # Attempt Pydantic direct parse first
    try:
        spec = RubricSpec.model_validate(data)
        logger.info("Parsed rubric directly from structured data (%s)", source)
        return spec
    except Exception as exc:
        logger.warning("Direct RubricSpec parse failed (%s): %s", source, exc)

    # Manual best-effort parse with safe defaults for missing/null fields
    parts: list[QuestionPartSpec] = []
    for raw_part in data.get("question_parts", []):
        criteria = [
            Criterion(
                criterion_id=c.get("criterion_id") or f"c{i}",
                name=c.get("name") or "Unnamed",
                description=c.get("description") or "",
                max_marks=float(c.get("max_marks") or 0),
                required_evidence=c.get("required_evidence") or [],
                partial_credit_rules=c.get("partial_credit_rules") or [],
                common_misconceptions=c.get("common_misconceptions") or [],
                no_credit_conditions=c.get("no_credit_conditions") or [],
                provenance=source,
            )
            for i, c in enumerate(raw_part.get("criteria") or [])
        ]
        buckets = [
            BucketDefinition(
                bucket=b["bucket"],
                label=b.get("label") or b["bucket"],
                mark_range=tuple(b["mark_range"]),
                description=b.get("description") or "",
            )
            for b in (raw_part.get("bucket_definitions") or [])
            if b.get("bucket") and b.get("mark_range")
        ]
        parts.append(
            QuestionPartSpec(
                question_id=raw_part.get("question_id") or "Q?",
                part_id=raw_part.get("part_id") or "",
                max_marks=float(raw_part.get("max_marks") or 0),
                task_instruction=raw_part.get("task_instruction") or "",
                criteria=criteria,
                bucket_definitions=buckets,
            )
        )
    return RubricSpec(
        assessment_name=data.get("assessment_name", "Unknown"),
        needs_human_confirmation=data.get("needs_human_confirmation", False),
        question_parts=parts,
    )


def _rubric_from_llm(
    marking_scheme_text: str,
    rubric_text: str,
    instructions_text: str,
    client: OllamaClient,
    model: str,
    temperature: float,
) -> RubricSpec:
    """Use LLM to extract rubric structure from free text."""
    prompt_template = ""
    if _RUBRIC_COMPILER_PROMPT_PATH.exists():
        prompt_template = _RUBRIC_COMPILER_PROMPT_PATH.read_text(encoding="utf-8")

    if prompt_template:
        prompt = (
            prompt_template
            .replace("{marking_scheme_text}", marking_scheme_text[:16000])
            .replace("{rubric_text}", rubric_text[:4000])
            .replace("{instructions_text}", instructions_text[:2000])
        )
    else:
        prompt = (
            "You are a rubric extraction assistant. Read the marking scheme below and extract "
            "a structured rubric for EVERY question part found.\n\n"
            "Return a JSON object with this exact structure:\n"
            '{\n'
            '  "assessment_name": "...",\n'
            '  "needs_human_confirmation": true,\n'
            '  "question_parts": [\n'
            '    {\n'
            '      "question_id": "Part_1",\n'
            '      "part_id": "",\n'
            '      "max_marks": 15,\n'
            '      "task_instruction": "Brief description of what students must do",\n'
            '      "criteria": [\n'
            '        {\n'
            '          "criterion_id": "c1",\n'
            '          "name": "...",\n'
            '          "description": "...",\n'
            '          "max_marks": 5,\n'
            '          "required_evidence": ["..."],\n'
            '          "partial_credit_rules": ["..."],\n'
            '          "common_misconceptions": ["..."],\n'
            '          "no_credit_conditions": ["..."],\n'
            '          "provenance": "marking_scheme"\n'
            '        }\n'
            '      ],\n'
            '      "bucket_definitions": [\n'
            '        {"bucket": "A", "label": "Excellent", "mark_range": [13, 15], "description": "..."},\n'
            '        {"bucket": "B", "label": "Strong",    "mark_range": [10, 12], "description": "..."},\n'
            '        {"bucket": "C", "label": "Competent", "mark_range": [7, 9],   "description": "..."},\n'
            '        {"bucket": "D", "label": "Limited",   "mark_range": [4, 6],   "description": "..."},\n'
            '        {"bucket": "E", "label": "Weak",      "mark_range": [1, 3],   "description": "..."},\n'
            '        {"bucket": "F", "label": "No credit", "mark_range": [0, 0],   "description": "..."}\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}\n\n'
            "IMPORTANT: Extract ALL parts from the marking scheme (Part 1, Part 2, Part 3, Part 4). "
            "For Part 2 which has numbered sub-questions, create one entry per sub-question "
            "(e.g. question_id=Part_2, part_id=q1 for sub-question 1). "
            "Derive bucket mark ranges from the part's max_marks. "
            "Return ONLY valid JSON, no other text.\n\n"
            f"MARKING SCHEME:\n{marking_scheme_text[:16000]}\n"
        )

    logger.info("Calling LLM to extract rubric from free text")
    raw = client.generate(model=model, prompt=prompt, temperature=temperature)

    # Extract JSON block if embedded in prose
    json_start = raw.find("{")
    json_end = raw.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        raw = raw[json_start:json_end]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM rubric extraction returned invalid JSON — retrying with explicit instruction")
        retry_prompt = prompt + "\n\nIMPORTANT: Your response MUST be valid JSON only. No other text."
        raw2 = client.generate(model=model, prompt=retry_prompt, temperature=0.0)
        j2 = raw2.find("{")
        j2_end = raw2.rfind("}") + 1
        if j2 >= 0:
            raw2 = raw2[j2:j2_end]
        try:
            data = json.loads(raw2)
        except json.JSONDecodeError as exc:
            logger.error("LLM rubric extraction failed after retry: %s", exc)
            # Return a minimal skeleton with confirmation flag set
            return RubricSpec(
                assessment_name="UNKNOWN — LLM extraction failed",
                needs_human_confirmation=True,
                question_parts=[],
            )

    spec = _rubric_from_structured(data, source="llm_extraction")
    spec.needs_human_confirmation = True  # Always flag AI-generated rubrics
    logger.warning("Rubric was AI-generated from free text — needs_human_confirmation=True")
    return spec


def run_rubric_compiler(config: PipelineConfig, force: bool = False) -> Path:
    """
    Compile rubric_spec.json from the files specified in *config*.

    Returns path to the written rubric_spec.json.
    """
    output_dir = Path(config.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rubric_spec.json"

    if output_path.exists() and not force:
        logger.info("rubric_spec.json already exists — skipping (use --force to overwrite)")
        return output_path

    marking_scheme_path = Path(config.marking_scheme_path)
    if not marking_scheme_path.exists():
        raise FileNotFoundError(f"Marking scheme not found: {marking_scheme_path}")

    marking_scheme_text = _load_file_text(marking_scheme_path)
    rubric_text = _load_file_text(Path(config.rubric_path)) if config.rubric_path else ""
    instructions_text = _load_file_text(Path(config.instructions_path)) if config.instructions_path else ""

    # Try structured parse of marking scheme first
    structured = _try_parse_structured(marking_scheme_text)
    if structured is not None:
        spec = _rubric_from_structured(structured, source=str(marking_scheme_path))
    else:
        # Free text — use LLM
        client = OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
            stage="rubric_compiler",
        )
        spec = _rubric_from_llm(
            marking_scheme_text=marking_scheme_text,
            rubric_text=rubric_text,
            instructions_text=instructions_text,
            client=client,
            model=config.models.analysis_model,
            temperature=config.temperature,
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(spec.model_dump_json(indent=2))

    if spec.needs_human_confirmation:
        logger.warning(
            "rubric_spec.json written but needs_human_confirmation=True. "
            "Please review %s before running scoring.",
            output_path,
        )
    else:
        logger.info("rubric_spec.json written to %s", output_path)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compile rubric_spec.json from marking scheme files")
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

    out = run_rubric_compiler(cfg, force=args.force)
    print(f"Rubric compiler complete: {out}")
    sys.exit(0)
