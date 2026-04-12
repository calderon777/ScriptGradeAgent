import io
import zipfile
from collections import Counter
from datetime import datetime

import pandas as pd


def build_results_workbook(df: pd.DataFrame) -> tuple[bytes, str]:
    output = io.BytesIO()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"ScriptGradeAgent_results_{timestamp}.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.getvalue(), excel_filename


def build_results_bundle(df: pd.DataFrame, report_text: str) -> tuple[bytes, str]:
    workbook_bytes, excel_filename = build_results_workbook(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ScriptGradeAgent_consistency_report_{timestamp}.md"
    bundle_filename = f"ScriptGradeAgent_results_bundle_{timestamp}.zip"
    zip_output = io.BytesIO()
    with zipfile.ZipFile(zip_output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(excel_filename, workbook_bytes)
        archive.writestr(report_filename, report_text)
    zip_output.seek(0)
    return zip_output.getvalue(), bundle_filename


def build_consistency_report(df: pd.DataFrame, selected_models: list[tuple[str, str]], max_mark: float) -> str:
    lines = [
        "# Consistency Report",
        "",
        f"Scripts processed: {len(df)}",
        f"Configured maximum mark: {max_mark:g}",
        "",
    ]

    for _, label in selected_models:
        status_col = f"{label}_status"
        error_col = f"{label}_error"
        mark_col = f"{label}_mark"
        feedback_col = f"{label}_feedback"
        part_count_col = f"{label}_detected_part_count"
        word_count_col = f"{label}_extracted_word_count"
        extraction_col = f"{label}_possible_extraction_issue"
        notes_col = f"{label}_validation_notes"
        calibration_col = f"{label}_calibration_delta"
        ai_total_col = f"{label}_ai_total_mark"
        math_total_col = f"{label}_math_total_mark"
        delta_col = f"{label}_ai_math_mark_delta"

        lines.extend([f"## {label}", ""])
        if status_col not in df.columns or mark_col not in df.columns:
            lines.extend(["No data available for this model.", ""])
            continue

        statuses = df[status_col].fillna("missing").astype(str)
        marks = pd.to_numeric(df[mark_col], errors="coerce")
        feedbacks = df[feedback_col].fillna("").astype(str) if feedback_col in df.columns else pd.Series(dtype=str)

        ok_count = int((statuses == "ok").sum())
        error_count = int((statuses == "error").sum())
        non_null_marks = marks.dropna()

        lines.append(f"Successful scripts: {ok_count}")
        lines.append(f"Failed scripts: {error_count}")

        if part_count_col in df.columns:
            part_counts = pd.to_numeric(df[part_count_col], errors="coerce").dropna()
            if not part_counts.empty:
                lines.append(f"Detected parts median: {part_counts.median():.0f}")
        if word_count_col in df.columns:
            word_counts = pd.to_numeric(df[word_count_col], errors="coerce").dropna()
            if not word_counts.empty:
                lines.append(f"Extracted words median: {word_counts.median():.0f}")
        if extraction_col in df.columns:
            extraction_issues = int(df[extraction_col].fillna(False).astype(bool).sum())
            lines.append(f"Possible extraction issues: {extraction_issues}")
        if calibration_col in df.columns:
            calibration_count = int(pd.to_numeric(df[calibration_col], errors="coerce").fillna(0).ne(0).sum())
            lines.append(f"Calibrated marks changed: {calibration_count}")
        if delta_col in df.columns:
            deltas = pd.to_numeric(df[delta_col], errors="coerce").dropna().abs()
            if not deltas.empty:
                lines.append(f"AI/math median delta: {deltas.median():.2f}")
                lines.append(f"AI/math max delta: {deltas.max():.2f}")
                lines.append(f"AI/math disagreements over 3 marks: {int(deltas.gt(3).sum())}")

        if non_null_marks.empty:
            lines.extend(["No marks available.", ""])
            continue

        top_mark_count = int((non_null_marks == max_mark).sum())
        unique_marks = sorted(non_null_marks.unique().tolist())
        lines.append(f"Marks returned: {len(non_null_marks)}")
        lines.append(f"Mean mark: {non_null_marks.mean():.2f}")
        lines.append(f"Median mark: {non_null_marks.median():.2f}")
        lines.append(f"Min / Max: {non_null_marks.min():.2f} / {non_null_marks.max():.2f}")
        lines.append(f"Distinct marks: {', '.join(_format_report_number(value) for value in unique_marks)}")
        lines.append(f"Top-mark rate: {top_mark_count}/{len(non_null_marks)} ({(top_mark_count / len(non_null_marks)):.1%})")

        distribution = non_null_marks.value_counts().sort_index()
        lines.extend(["", "### Mark Distribution", ""])
        for mark_value, count in distribution.items():
            lines.append(f"- {_format_report_number(float(mark_value))}: {int(count)}")

        repeated_starts = Counter(_feedback_signature(text) for text in feedbacks if text.strip())
        common_starts = [(phrase, count) for phrase, count in repeated_starts.most_common(5) if phrase]
        lines.extend(["", "### Repeated Feedback Starts", ""])
        if common_starts:
            for phrase, count in common_starts:
                lines.append(f"- {count}x: {phrase}")
        else:
            lines.append("- None detected")

        if error_count:
            error_messages = Counter(str(value).strip() for value in df[error_col].fillna("").astype(str) if str(value).strip())
            lines.extend(["", "### Error Summary", ""])
            for message, count in error_messages.most_common(5):
                lines.append(f"- {count}x: {message}")

        if notes_col in df.columns:
            note_values = Counter()
            for value in df[notes_col].fillna("").astype(str):
                for item in [part.strip() for part in value.split("|") if part.strip()]:
                    note_values[item] += 1
            if note_values:
                lines.extend(["", "### Validation Notes", ""])
                for message, count in note_values.most_common(5):
                    lines.append(f"- {count}x: {message}")

        if ai_total_col in df.columns and math_total_col in df.columns:
            ai_totals = pd.to_numeric(df[ai_total_col], errors="coerce")
            math_totals = pd.to_numeric(df[math_total_col], errors="coerce")
            disagreements = df.loc[
                ai_totals.notna() & math_totals.notna() & ai_totals.ne(math_totals),
                ["filename", ai_total_col, math_total_col],
            ]
            if not disagreements.empty:
                lines.extend(["", "### AI vs Math Total Differences", ""])
                for _, row in disagreements.head(10).iterrows():
                    lines.append(
                        f"- {row['filename']}: AI {_format_report_number(float(row[ai_total_col]))}, "
                        f"Math {_format_report_number(float(row[math_total_col]))}"
                    )

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_verifier_report(df: pd.DataFrame, verifier_label: str) -> str:
    agreement_col = f"{verifier_label}_agreement"
    confidence_col = f"{verifier_label}_confidence"
    issues_col = f"{verifier_label}_issues"
    recommendation_col = f"{verifier_label}_recommendation"
    status_col = f"{verifier_label}_status"
    error_col = f"{verifier_label}_error"

    if status_col not in df.columns:
        return ""

    lines = ["# Verifier Report", ""]
    statuses = df[status_col].fillna("missing").astype(str)
    lines.append(f"Verifier passes: {int((statuses == 'ok').sum())}")
    lines.append(f"Verifier failures: {int((statuses == 'error').sum())}")
    lines.append("")

    if agreement_col in df.columns:
        agreements = df[agreement_col].fillna("missing").astype(str)
        lines.append("## Agreement")
        lines.append("")
        for value, count in agreements.value_counts().items():
            lines.append(f"- {value}: {int(count)}")
        lines.append("")

    if confidence_col in df.columns:
        confidences = pd.to_numeric(df[confidence_col], errors="coerce").dropna()
        if not confidences.empty:
            lines.append(f"Median verifier confidence: {confidences.median():.0f}")
            lines.append("")

    if recommendation_col in df.columns:
        lines.append("## Recommendations")
        lines.append("")
        recommendations = Counter(str(value).strip() for value in df[recommendation_col].fillna("").astype(str) if str(value).strip())
        for message, count in recommendations.most_common(5):
            lines.append(f"- {count}x: {message}")
        lines.append("")

    if issues_col in df.columns:
        lines.append("## Common Issues")
        lines.append("")
        issue_counts = Counter()
        for value in df[issues_col].fillna("").astype(str):
            for item in [part.strip() for part in value.split("|") if part.strip()]:
                issue_counts[item] += 1
        for message, count in issue_counts.most_common(5):
            lines.append(f"- {count}x: {message}")
        lines.append("")

    if error_col in df.columns:
        verifier_errors = Counter(str(value).strip() for value in df[error_col].fillna("").astype(str) if str(value).strip())
        if verifier_errors:
            lines.append("## Verifier Errors")
            lines.append("")
            for message, count in verifier_errors.most_common(5):
                lines.append(f"- {count}x: {message}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _feedback_signature(text: str) -> str:
    words = str(text).split()
    return " ".join(words[:12])


def _format_report_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"
