import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from marking_pipeline import (
    DEFAULT_CONTEXT_PATTERNS,
    build_document_based_marking_text,
    build_marking_context_with_optional_override,
    build_rubric_matrix_markdown,
    build_single_assessment_bundle,
    combine_text_sections,
    read_paths_text,
    save_ingest_snapshot,
)


@dataclass
class LocalUpload:
    name: str
    source_path: Path

    def getvalue(self) -> bytes:
        return self.source_path.read_bytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and export the rubric matrix without running grading.")
    parser.add_argument("--assessment-root", required=True, help="Folder containing student subfolders for one assessment.")
    parser.add_argument("--marking-scheme", action="append", default=[], help="Marking document path. May be passed multiple times.")
    parser.add_argument("--max-mark", type=float, required=True, help="Configured overall maximum mark.")
    parser.add_argument("--document-only", action="store_true", help="Use uploaded documents when there is no rubric.")
    parser.add_argument("--output-dir", default="output", help="Output directory for rubric artifacts.")
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

    rubric_markdown = build_rubric_matrix_markdown(context, assessment_name=bundle.name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{bundle.name}_rubric_matrix.md"
    xlsx_path = output_dir / f"{bundle.name}_rubric_matrix.xlsx"
    md_path.write_text(rubric_markdown, encoding="utf-8")
    build_rubric_workbook(rubric_markdown, xlsx_path)
    print(f"Saved rubric markdown to {md_path}")
    print(f"Saved rubric workbook to {xlsx_path}")


def build_rubric_workbook(rubric_markdown: str, output_path: Path) -> None:
    records = parse_rubric_matrix_markdown(rubric_markdown)
    raw_df = pd.DataFrame({"rubric_markdown": rubric_markdown.splitlines()})
    matrix_df = pd.DataFrame(records)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_markdown", index=False)
        if not matrix_df.empty:
            matrix_df.to_excel(writer, sheet_name="rubric_matrix", index=False)


def parse_rubric_matrix_markdown(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_section = ""
    current_unit = ""
    max_mark = ""
    mode = ""
    classification_question = ""
    ranking_rule = ""
    criteria: list[str] = []
    in_table = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_section = stripped[3:].strip()
            current_unit = ""
            criteria = []
            in_table = False
            continue
        if stripped.startswith("### "):
            current_unit = stripped[4:].strip()
            max_mark = ""
            mode = ""
            classification_question = ""
            ranking_rule = ""
            criteria = []
            in_table = False
            continue
        if stripped.startswith("- Max mark:"):
            max_mark = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("- Mode:"):
            mode = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("- Classification question:"):
            classification_question = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("- Ranking rule:"):
            ranking_rule = stripped.split(":", 1)[1].strip()
            continue
        if stripped == "#### Criteria Sentences":
            in_table = False
            continue
        if (
            stripped.startswith("- ")
            and current_unit
            and not stripped.startswith("- Max mark:")
            and not stripped.startswith("- Mode:")
            and not stripped.startswith("- Dependency group:")
            and not stripped.startswith("- Core criterion:")
            and not stripped.startswith("- Classification question:")
            and not stripped.startswith("- Ranking rule:")
        ):
            criteria.append(stripped[2:].strip())
            continue
        if stripped.startswith("| Band | Descriptor | Within-band placement |"):
            in_table = True
            continue
        if in_table and stripped.startswith("| ---"):
            continue
        if in_table and stripped.startswith("|"):
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) >= 3:
                records.append(
                    {
                        "section": current_section,
                        "unit": current_unit,
                        "max_mark": max_mark,
                        "mode": mode,
                        "classification_question": classification_question,
                        "ranking_rule": ranking_rule,
                        "criteria_sentences": "\n".join(criteria),
                        "band": parts[0],
                        "descriptor": parts[1],
                        "within_band_placement": parts[2],
                    }
                )
    return records


def snapshot_marking_scheme_paths(snapshot_path: Path) -> list[str]:
    import json

    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return manifest.get("documents", {}).get("marking_scheme_files", [])


if __name__ == "__main__":
    main()
