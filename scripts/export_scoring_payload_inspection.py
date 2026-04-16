import argparse
import json
import sys
from pathlib import Path

from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_human_benchmark import DEFAULT_SAMPLE_PATHS, MODEL_NAME, apply_variant_to_parts, load_context
from marking_pipeline.core import (
    _build_part_task_text,
    build_part_messages,
    build_submission_texts_from_path,
    detect_submission_parts,
    prepare_assessment_map,
    refine_part_task_types_with_model,
    refine_submission_granularity,
    segment_submission_parts,
    DEFAULT_OLLAMA_URL,
)


def _extract_payload_from_messages(messages: list[dict[str, str]]) -> dict:
    user_prompt = messages[1]["content"]
    prefix = "scoring_payload:\n"
    if not user_prompt.startswith(prefix):
        raise ValueError("Unexpected prompt format: scoring payload prefix missing.")
    return json.loads(user_prompt[len(prefix):].strip())


def _parent_label_for_part(label: str) -> str:
    if " Q" not in label:
        return ""
    return label.rsplit(" Q", 1)[0].strip()


def _render_payload_marking_guidance(payload: dict) -> str:
    scoring = payload.get("scoring", {})
    criteria = scoring.get("criteria", [])
    rules = scoring.get("rules", [])
    scope_block = scoring.get("scope_block", [])
    decision_block = scoring.get("decision_block", [])
    output_block = scoring.get("output_block", [])
    lines: list[str] = []
    if criteria:
        lines.append("Payload criteria:")
        lines.extend(
            f"- {str(item.get('check', '')).strip()}"
            for item in criteria
            if str(item.get("check", "")).strip()
        )
    if rules:
        lines.append("Payload scoring rules:")
        lines.extend(f"- {str(rule).strip()}" for rule in rules if str(rule).strip())
    if scope_block:
        lines.append("Payload scope block:")
        lines.extend(f"- {str(rule).strip()}" for rule in scope_block if str(rule).strip())
    if decision_block:
        lines.append("Payload decision block:")
        lines.extend(f"- {str(rule).strip()}" for rule in decision_block if str(rule).strip())
    if output_block:
        lines.append("Payload output block:")
        lines.extend(f"- {str(rule).strip()}" for rule in output_block if str(rule).strip())
    return "\n".join(lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the exact per-pass scoring payload to Excel for inspection.")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATHS[0]), help="Submission path to inspect.")
    parser.add_argument(
        "--variant",
        choices=("cached_map_only", "prepared_artifact_enriched"),
        default="prepared_artifact_enriched",
        help="Payload variant to inspect.",
    )
    parser.add_argument(
        "--output",
        default=str(Path.home() / "Downloads" / "scoring_pass_payload_inspection.xlsx"),
        help="Output Excel workbook path.",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Optional verifier model used when preparing the assessment map.",
    )
    parser.add_argument(
        "--task-type-model",
        default=None,
        help="Optional model used to refine ambiguous task_type assignments, for example mistral:7b.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context = load_context()
    prepared_assessment_map = prepare_assessment_map(
        context=context,
        verifier_model_name=args.verifier_model,
    )
    script_text, structure_text = build_submission_texts_from_path(sample_path)
    structure_input_text = structure_text or script_text

    parts = detect_submission_parts(
        structure_input_text,
        sample_path.name,
        MODEL_NAME,
        context=context,
    )
    parts = segment_submission_parts(script_text, parts)
    top_level_max_by_label = {part.label: part.max_mark for part in parts}
    parts = refine_submission_granularity(parts, context, sample_path.name, MODEL_NAME, include_unmarked_subparts=True)
    if args.task_type_model:
        parts = refine_part_task_types_with_model(parts, args.task_type_model, ollama_url=DEFAULT_OLLAMA_URL)
    parts = apply_variant_to_parts(parts, prepared_assessment_map, args.variant)

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    prompt_sheet = workbook.create_sheet("Prompts")

    summary_headers = [
        "index",
        "label",
        "parent_label",
        "max_mark",
        "parent_max_mark",
        "task_type",
        "criterion_mode",
        "section_word_count",
        "task_text",
        "criteria_count",
        "criterion_1",
        "criterion_2",
        "criterion_3",
        "criterion_4",
        "criterion_5",
        "criterion_scale_type",
        "criterion_scale_labels",
        "scoring_rules",
        "question_text_exact",
        "source_marking_guidance",
        "payload_marking_guidance",
        "section_text",
    ]
    prompt_headers = ["index", "label", "system_prompt", "user_prompt"]
    summary_sheet.append(summary_headers)
    prompt_sheet.append(prompt_headers)

    for index, part in enumerate(parts, start=1):
        task_text = _build_part_task_text(part)
        messages = build_part_messages(part, context, sample_path.name, model_name=MODEL_NAME)
        payload = _extract_payload_from_messages(messages)
        parent_label = _parent_label_for_part(part.label)
        parent_max_mark = top_level_max_by_label.get(parent_label) if parent_label else None
        payload_criteria = payload.get("scoring", {}).get("criteria", [])
        criteria = [str(item.get("check", "")).strip() for item in payload_criteria if str(item.get("check", "")).strip()]
        scoring_rules = [
            str(item).strip()
            for item in payload.get("scoring", {}).get("rules", [])
            if str(item).strip()
        ]
        scale_payload = payload.get("scoring", {}).get("scale", {})
        first_scale_type = str(scale_payload.get("type", "")).strip()
        first_scale_labels = ", ".join(str(item).strip() for item in scale_payload.get("labels", []) if str(item).strip())
        payload_marking_guidance = _render_payload_marking_guidance(payload)
        summary_sheet.append(
            [
                index,
                part.label,
                parent_label,
                part.max_mark,
                parent_max_mark,
                part.task_type,
                part.criterion_mode,
                len(part.section_text.split()),
                task_text,
                len(criteria),
                *(criteria + [""] * (5 - len(criteria))),
                first_scale_type,
                first_scale_labels,
                "\n".join(scoring_rules),
                part.question_text_exact,
                part.marking_guidance,
                payload_marking_guidance,
                part.section_text,
            ]
        )
        prompt_sheet.append(
            [
                index,
                part.label,
                messages[0]["content"],
                messages[1]["content"],
            ]
        )

    for sheet in (summary_sheet, prompt_sheet):
        sheet.freeze_panes = "A2"
        for column in sheet.columns:
            letter = column[0].column_letter
            if letter in {"A", "B", "C", "D", "E", "G"}:
                continue
            sheet.column_dimensions[letter].width = 30

    workbook.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
