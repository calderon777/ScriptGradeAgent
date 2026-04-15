import argparse
import sys
from pathlib import Path

from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_human_benchmark import DEFAULT_SAMPLE_PATHS, MODEL_NAME, apply_variant_to_parts, load_context
from marking_pipeline.core import (
    _build_part_task_text,
    _extract_prompt_criteria,
    _extract_prompt_scoring_rules,
    build_part_messages,
    build_submission_texts_from_path,
    detect_submission_parts,
    prepare_assessment_map,
    refine_submission_granularity,
    segment_submission_parts,
)


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
    parts = refine_submission_granularity(parts, context, sample_path.name, MODEL_NAME)
    parts = apply_variant_to_parts(parts, prepared_assessment_map, args.variant)

    workbook = Workbook()
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    prompt_sheet = workbook.create_sheet("Prompts")

    summary_headers = [
        "index",
        "label",
        "max_mark",
        "task_type",
        "section_word_count",
        "task_text",
        "criteria_count",
        "criterion_1",
        "criterion_2",
        "criterion_3",
        "criterion_4",
        "criterion_5",
        "scoring_rules",
        "question_text_exact",
        "marking_guidance",
        "section_text",
    ]
    prompt_headers = ["index", "label", "system_prompt", "user_prompt"]
    summary_sheet.append(summary_headers)
    prompt_sheet.append(prompt_headers)

    for index, part in enumerate(parts, start=1):
        task_text = _build_part_task_text(part)
        criteria = _extract_prompt_criteria(part.marking_guidance)
        scoring_rules = _extract_prompt_scoring_rules(part.marking_guidance, part.max_mark)
        messages = build_part_messages(part, context, sample_path.name)
        summary_sheet.append(
            [
                index,
                part.label,
                part.max_mark,
                part.task_type,
                len(part.section_text.split()),
                task_text,
                len(criteria),
                *(criteria + [""] * (5 - len(criteria))),
                "\n".join(scoring_rules),
                part.question_text_exact,
                part.marking_guidance,
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
