import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marking_pipeline import (
    DEFAULT_CONTEXT_PATTERNS,
    apply_model_result,
    apply_verifier_result,
    build_consistency_report,
    build_document_based_marking_text,
    build_marking_context_with_optional_override,
    build_results_bundle,
    build_single_assessment_bundle,
    call_ollama,
    combine_text_sections,
    maybe_run_regrade_loop,
    read_path_text,
    read_paths_text,
    save_ingest_snapshot,
)


@dataclass
class LocalUpload:
    name: str
    source_path: Path

    def getvalue(self) -> bytes:
        return self.source_path.read_bytes()


AVAILABLE_MODELS = {
    "Qwen2 7B": ("qwen2:7b", "Qwen2_7B"),
    "Gemma 3 4B": ("gemma3:4b", "Gemma3_4B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save an ingest snapshot and run a warm-test batch.")
    parser.add_argument("--assessment-root", required=True, help="Folder containing student subfolders for one assessment.")
    parser.add_argument("--marking-scheme", action="append", default=[], help="Marking document path. May be passed multiple times.")
    parser.add_argument("--max-mark", type=float, required=True, help="Configured overall maximum mark.")
    parser.add_argument("--limit", type=int, default=5, help="Number of scripts to run.")
    parser.add_argument("--grader", default="Qwen2 7B", choices=AVAILABLE_MODELS.keys())
    parser.add_argument("--verifier", default="Gemma 3 4B", choices=AVAILABLE_MODELS.keys())
    parser.add_argument("--document-only", action="store_true", help="Use uploaded documents when there is no rubric.")
    parser.add_argument("--output-dir", default="output", help="Output directory for workbook bundle.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assessment_root = Path(args.assessment_root).expanduser()
    marking_scheme_paths = [Path(item).expanduser() for item in args.marking_scheme]
    if not assessment_root.exists():
        raise SystemExit(f"Assessment root not found: {assessment_root}")
    for path in marking_scheme_paths:
        if not path.exists():
            raise SystemExit(f"Marking document not found: {path}")

    snapshot_path = save_ingest_snapshot(
        use_assessment_folders=True,
        single_assessment_folder_mode=True,
        assessment_root=str(assessment_root),
        folder_keywords={
            "rubric": DEFAULT_CONTEXT_PATTERNS["rubric"],
            "brief": DEFAULT_CONTEXT_PATTERNS["brief"],
            "marking_scheme": DEFAULT_CONTEXT_PATTERNS["marking_scheme"],
            "graded_sample": DEFAULT_CONTEXT_PATTERNS["graded_sample"],
            "other": DEFAULT_CONTEXT_PATTERNS["other"],
        },
        scale_profile=f"Custom {args.max_mark:g}",
        manual_max_mark=float(args.max_mark),
        document_only_mode=bool(args.document_only),
        script_files=[],
        csv_file=None,
        rubric_files=[],
        brief_files=[],
        marking_scheme_files=[LocalUpload(path.name, path) for path in marking_scheme_paths],
        graded_sample_files=[],
        other_files=[],
    )
    print(f"Saved ingest snapshot to {snapshot_path}")

    bundle = build_single_assessment_bundle(
        assessment_root,
        assessment_name=assessment_root.name,
        rubric_keywords=DEFAULT_CONTEXT_PATTERNS["rubric"],
        brief_keywords=DEFAULT_CONTEXT_PATTERNS["brief"],
        marking_scheme_keywords=DEFAULT_CONTEXT_PATTERNS["marking_scheme"],
        graded_sample_keywords=DEFAULT_CONTEXT_PATTERNS["graded_sample"],
        other_keywords=DEFAULT_CONTEXT_PATTERNS["other"],
    )

    uploaded_marking_scheme_text = read_paths_text([Path(path) for path in snapshot_marking_scheme_paths(snapshot_path)])
    rubric_text = read_paths_text(bundle.rubric_files)
    brief_text = read_paths_text(bundle.brief_files)
    marking_scheme_text = combine_text_sections(read_paths_text(bundle.marking_scheme_files), uploaded_marking_scheme_text)
    graded_sample_text = read_paths_text(bundle.graded_sample_files)
    other_context_text = read_paths_text(bundle.other_files)

    effective_rubric_text = rubric_text
    if not effective_rubric_text.strip() and args.document_only:
        effective_rubric_text = build_document_based_marking_text(
            brief_text=brief_text,
            marking_scheme_text=marking_scheme_text,
            graded_sample_text=graded_sample_text,
            other_context_text=other_context_text,
        )

    context = build_marking_context_with_optional_override(
        rubric_text=effective_rubric_text,
        brief_text=brief_text,
        marking_scheme_text=marking_scheme_text,
        graded_sample_text=graded_sample_text,
        other_context_text=other_context_text,
        manual_max_mark=float(args.max_mark),
    )

    grader_model = AVAILABLE_MODELS[args.grader]
    verifier_model = AVAILABLE_MODELS[args.verifier]

    rows: list[dict[str, object]] = []
    submission_paths = list(bundle.submission_files)[: args.limit]
    if not submission_paths:
        raise SystemExit("No submission files found in the assessment root.")

    for index, submission_path in enumerate(submission_paths, start=1):
        print(f"[{index}/{len(submission_paths)}] {submission_path.name}")
        row: dict[str, object] = {
            "assessment_name": bundle.name,
            "filename": submission_path.name,
            "source_path": str(submission_path),
        }
        try:
            script_text = read_path_text(submission_path)
        except Exception as exc:
            apply_model_result(row, grader_model[1], error=exc)
            apply_verifier_result(row, verifier_model[1], error=exc)
            rows.append(row)
            print(f"  read error: {exc}")
            continue

        try:
            result = call_ollama(
                script_text=script_text,
                context=context,
                filename=submission_path.name,
                model_name=grader_model[0],
            )
            apply_model_result(row, grader_model[1], result=result)
            print(f"  grader: {result['total_mark']}/{result['max_mark']}")
        except Exception as exc:
            apply_model_result(row, grader_model[1], error=exc)
            print(f"  grader error: {exc}")

        maybe_run_regrade_loop(
            row=row,
            script_text=script_text,
            context=context,
            filename=submission_path.name,
            grader_model=grader_model,
            verifier_model=verifier_model,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    report_text = build_consistency_report(df, [grader_model], context.max_mark)
    bundle_bytes, bundle_name = build_results_bundle(df, report_text)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / bundle_name
    output_path.write_bytes(bundle_bytes)
    print(f"Saved results bundle to {output_path}")


def snapshot_marking_scheme_paths(snapshot_path: Path) -> list[str]:
    import json

    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return manifest.get("documents", {}).get("marking_scheme_files", [])


if __name__ == "__main__":
    main()
