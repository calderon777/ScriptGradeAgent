import io
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

from marking_pipeline import MarkingContext, call_ollama, prepare_marking_context, read_uploaded_files_text


AVAILABLE_MODELS = {
    "Llama 3.1 8B": ("llama3.1:8b", "Llama3.1_8B"),
    "Mistral 7B": ("mistral:7b", "Mistral_7B"),
    "Qwen2 7B": ("qwen2:7b", "Qwen2_7B"),
    "Gemma 3 4B": ("gemma3:4b", "Gemma3_4B"),
}


def build_csv_script_text(row_data: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = row_data.get(column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{column}:\n{text}")
    return "\n\n".join(parts).strip()


def build_document_based_marking_text(
    brief_text: str,
    marking_scheme_text: str,
    graded_sample_text: str,
    other_context_text: str,
) -> str:
    sections: list[str] = []
    if marking_scheme_text.strip():
        sections.append(f"MARKING SCHEME:\n{marking_scheme_text.strip()}")
    if brief_text.strip():
        sections.append(f"ASSIGNMENT BRIEF:\n{brief_text.strip()}")
    if graded_sample_text.strip():
        sections.append(f"EXAMPLE GRADED SCRIPT:\n{graded_sample_text.strip()}")
    if other_context_text.strip():
        sections.append(f"OTHER SUPPORTING DOCUMENTS:\n{other_context_text.strip()}")
    return "\n\n".join(sections).strip()


def build_marking_context_with_optional_override(
    rubric_text: str,
    brief_text: str,
    marking_scheme_text: str,
    graded_sample_text: str,
    other_context_text: str,
    manual_max_mark: float | None,
) -> MarkingContext:
    try:
        return prepare_marking_context(
            rubric_text=rubric_text,
            brief_text=brief_text,
            marking_scheme_text=marking_scheme_text,
            graded_sample_text=graded_sample_text,
            other_context_text=other_context_text,
        )
    except ValueError as exc:
        if manual_max_mark is None or "Could not infer the maximum mark" not in str(exc):
            raise

    if not (rubric_text.strip() or marking_scheme_text.strip()):
        raise ValueError("Provide a rubric or marking scheme before grading.")

    return MarkingContext(
        rubric_text=rubric_text.strip(),
        brief_text=brief_text.strip(),
        marking_scheme_text=marking_scheme_text.strip(),
        graded_sample_text=graded_sample_text.strip(),
        other_context_text=other_context_text.strip(),
        max_mark=float(manual_max_mark),
    )


def render_results_download(df: pd.DataFrame) -> None:
    output = io.BytesIO()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"ScriptGradeAgent_results_{timestamp}.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    st.download_button(
        label="Download Excel file",
        data=output,
        file_name=excel_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_consistency_report(report_text: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ScriptGradeAgent_consistency_report_{timestamp}.md"
    st.subheader("Consistency Report")
    st.code(report_text, language="markdown")
    st.download_button(
        label="Download consistency report",
        data=report_text,
        file_name=filename,
        mime="text/markdown",
    )


def apply_model_result(row: dict[str, object], label: str, result: dict[str, object] | None = None, error: Exception | None = None) -> None:
    row[f"{label}_status"] = "ok" if error is None else "error"
    row[f"{label}_error"] = "" if error is None else str(error)
    if result is None:
        row[f"{label}_mark"] = None
        row[f"{label}_feedback"] = ""
        return
    row[f"{label}_mark"] = result["total_mark"]
    row[f"{label}_feedback"] = result["overall_feedback"]


def _feedback_signature(text: str) -> str:
    words = str(text).split()
    return " ".join(words[:12])


def _format_report_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def main() -> None:
    st.set_page_config(page_title="ScriptGradeAgent", layout="wide")

    with st.sidebar:
        st.title("ScriptGradeAgent")
        sidebar_status = st.empty()
        sidebar_status.info("Ready")

    st.title("ScriptGradeAgent")
    st.markdown(
        "Upload student scripts or a CSV of text answers, add a rubric or marking scheme, "
        "and generate structured marks and feedback with local Ollama models."
    )

    st.header("Step 1 - Models")
    st.markdown("Select one or more local models. Make sure Ollama is running and the models are already pulled.")
    for display_name, (model_name, _) in AVAILABLE_MODELS.items():
        st.code(f"ollama pull {model_name}", language="bash")

    selected_model_keys = st.multiselect(
        "Models to run",
        options=list(AVAILABLE_MODELS.keys()),
        default=["Llama 3.1 8B"],
    )
    selected_models = [AVAILABLE_MODELS[key] for key in selected_model_keys]

    st.header("Step 2 - Student Submissions")
    script_files = st.file_uploader(
        "Upload student scripts (.pdf or .docx)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.subheader("Or upload a CSV with text answers")
    csv_file = st.file_uploader(
        "CSV file with one row per student",
        type=["csv"],
        key="csv_assessments",
    )

    df_csv_preview = None
    if csv_file is not None:
        try:
            csv_file.seek(0)
            df_csv_preview = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            csv_file.seek(0)
            df_csv_preview = pd.read_csv(csv_file, encoding="cp1252")

        st.caption("Preview")
        st.dataframe(df_csv_preview.head(), use_container_width=True)

        columns = list(df_csv_preview.columns)
        default_id_index = columns.index("Id") if "Id" in columns else 0
        st.selectbox(
            "Column to use as the submission identifier",
            options=columns,
            index=default_id_index,
            key="csv_id_col",
        )

        default_text_columns = [column for column in columns if "question" in column.lower() or df_csv_preview[column].dtype == object]
        if not default_text_columns:
            default_text_columns = columns

        st.multiselect(
            "Columns containing answers to mark",
            options=columns,
            default=default_text_columns,
            key="csv_text_cols",
        )

    st.header("Step 3 - Marking Documents")
    st.caption("Upload a rubric when you have one. If not, you can tell the app to grade from the uploaded marking documents instead.")

    rubric_files = st.file_uploader(
        "Rubric files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="rubric_files",
    )
    brief_files = st.file_uploader(
        "Assignment brief files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="brief_files",
    )
    marking_scheme_files = st.file_uploader(
        "Marking scheme files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="marking_scheme_files",
    )
    graded_sample_files = st.file_uploader(
        "Example graded script files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="graded_sample_files",
    )
    other_files = st.file_uploader(
        "Other supporting documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="other_files",
    )

    rubric_text = read_uploaded_files_text(rubric_files)
    brief_text = read_uploaded_files_text(brief_files)
    marking_scheme_text = read_uploaded_files_text(marking_scheme_files)
    graded_sample_text = read_uploaded_files_text(graded_sample_files)
    other_context_text = read_uploaded_files_text(other_files)

    if "document_only_mode" not in st.session_state:
        st.session_state["document_only_mode"] = False

    if st.button("Use Uploaded Documents Instead of a Rubric", key="document_only_button"):
        st.session_state["document_only_mode"] = True

    if st.session_state["document_only_mode"]:
        st.info(
            "Document-only grading is enabled. The app will use the uploaded marking scheme, brief, "
            "example grading, and other supporting documents as the grading basis when no rubric is provided."
        )

    manual_max_mark_enabled = st.checkbox(
        "Enter maximum mark manually",
        value=False,
        help="Use this when the uploaded documents do not state the total mark in a way the app can detect.",
    )
    manual_max_mark = None
    if manual_max_mark_enabled:
        manual_max_mark = st.number_input(
            "Maximum mark",
            min_value=1.0,
            step=1.0,
            value=100.0,
        )

    st.header("Step 4 - Run Marking")
    if st.button("Run marking", type="primary"):
        if not selected_models:
            st.error("Select at least one model before running the marking.")
            return
        if not script_files and csv_file is None:
            st.error("Upload at least one script or a CSV file.")
            return

        try:
            effective_rubric_text = rubric_text
            if not effective_rubric_text.strip() and st.session_state.get("document_only_mode", False):
                effective_rubric_text = build_document_based_marking_text(
                    brief_text=brief_text,
                    marking_scheme_text=marking_scheme_text,
                    graded_sample_text=graded_sample_text,
                    other_context_text=other_context_text,
                )

            marking_context = build_marking_context_with_optional_override(
                rubric_text=effective_rubric_text,
                brief_text=brief_text,
                marking_scheme_text=marking_scheme_text,
                graded_sample_text=graded_sample_text,
                other_context_text=other_context_text,
                manual_max_mark=manual_max_mark,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        sidebar_status.info(f"Marking in progress ({marking_context.max_mark:g}-point scale)")
        rows: list[dict[str, object]] = []
        progress = st.progress(0)
        status = st.empty()

        if csv_file is not None:
            try:
                csv_file.seek(0)
                try:
                    df_input = pd.read_csv(csv_file)
                except UnicodeDecodeError:
                    csv_file.seek(0)
                    df_input = pd.read_csv(csv_file, encoding="cp1252")
            except Exception as exc:
                st.error(f"Error reading CSV file: {exc}")
                return

            columns = list(df_input.columns)
            csv_id_col = st.session_state.get("csv_id_col", columns[0])
            csv_text_cols = st.session_state.get("csv_text_cols", columns)
            total = len(df_input.index)

            for index, (_, row_data) in enumerate(df_input.iterrows(), start=1):
                filename = str(row_data.get(csv_id_col, f"row_{index}"))
                script_text = build_csv_script_text(row_data, csv_text_cols)
                status.text(f"Marking {filename} ({index}/{total})")

                row: dict[str, object] = {"filename": filename}
                for model_name, label in selected_models:
                    try:
                        result = call_ollama(
                            script_text=script_text,
                            context=marking_context,
                            filename=filename,
                            model_name=model_name,
                        )
                        apply_model_result(row, label, result=result)
                    except Exception as exc:
                        apply_model_result(row, label, error=exc)

                rows.append(row)
                progress.progress(index / total)

            df_marks = pd.DataFrame(rows)
            try:
                df_input_copy = df_input.copy()
                df_input_copy["_scriptgrade_id"] = df_input_copy[csv_id_col].astype(str)
                df_marks["_scriptgrade_id"] = df_marks["filename"].astype(str)
                df = df_input_copy.merge(
                    df_marks.drop(columns=["filename"]),
                    on="_scriptgrade_id",
                    how="left",
                ).drop(columns=["_scriptgrade_id"])
            except Exception:
                df = df_marks
        else:
            total = len(script_files)
            for index, uploaded in enumerate(script_files, start=1):
                filename = uploaded.name
                status.text(f"Marking {filename} ({index}/{total})")
                script_text = read_uploaded_files_text([uploaded])

                row = {"filename": filename}
                for model_name, label in selected_models:
                    try:
                        result = call_ollama(
                            script_text=script_text,
                            context=marking_context,
                            filename=filename,
                            model_name=model_name,
                        )
                        apply_model_result(row, label, result=result)
                    except Exception as exc:
                        apply_model_result(row, label, error=exc)

                rows.append(row)
                progress.progress(index / total)

            df = pd.DataFrame(rows)

        if df.empty:
            st.warning("No results to show.")
            sidebar_status.info("Ready")
            return

        mark_columns = [column for column in df.columns if column.endswith("_mark")]
        if len(mark_columns) >= 2:
            df["average_mark"] = df[mark_columns].mean(axis=1)
            if len(mark_columns) == 2:
                df["mark_difference"] = (df[mark_columns[0]] - df[mark_columns[1]]).abs()

        st.success("Marking completed.")
        sidebar_status.success("Completed")
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        render_results_download(df)
        consistency_report = build_consistency_report(df, selected_models, marking_context.max_mark)
        render_consistency_report(consistency_report)


if __name__ == "__main__":
    main()
