from pathlib import Path
from tkinter import Tk, filedialog

import pandas as pd
import streamlit as st

from marking_pipeline import (
    DEFAULT_CONTEXT_PATTERNS,
    SCALE_PRESETS,
    AssessmentBundle,
    MarkingContext,
    build_single_assessment_bundle,
    build_consistency_report,
    build_document_based_marking_text,
    build_marking_context_with_optional_override,
    build_results_bundle,
    build_rubric_matrix_markdown,
    build_verifier_report,
    call_ollama,
    build_submission_texts_from_path,
    build_submission_texts_from_upload,
    combine_text_sections,
    discover_assessment_bundles,
    get_last_ingest_manifest_path,
    load_ingest_snapshot,
    parse_keyword_patterns,
    read_path_text,
    read_paths_text,
    read_uploaded_files_text,
    save_ingest_snapshot,
    suggest_marking_scale,
    apply_cross_student_calibration,
    apply_model_result,
    apply_verifier_result,
    maybe_run_regrade_loop,
)


AVAILABLE_MODELS = {
    "Qwen2 7B": ("qwen2:7b", "Qwen2_7B"),
    "Gemma 3 4B": ("gemma3:4b", "Gemma3_4B"),
    "Llama 3.1 8B": ("llama3.1:8b", "Llama3.1_8B"),
    "Mistral 7B": ("mistral:7b", "Mistral_7B"),
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


def render_consistency_report(report_text: str) -> None:
    st.subheader("Consistency Report")
    st.code(report_text, language="markdown")


def render_rubric_matrix(rubric_text: str) -> None:
    if not rubric_text.strip():
        return
    st.subheader("Rubric Matrix")
    st.code(rubric_text, language="markdown")


def render_results_bundle_download(df: pd.DataFrame, report_text: str, rubric_text: str = "") -> None:
    bundle_bytes, bundle_filename = build_results_bundle(df, report_text, rubric_text=rubric_text)
    st.download_button(
        label="Download marksheet, consistency report, and rubric",
        data=bundle_bytes,
        file_name=bundle_filename,
        mime="application/zip",
    )


def build_bundle_summary_rows(bundles: list[AssessmentBundle]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        rows.append(
            {
                "assessment": bundle.name,
                "submissions": len(bundle.submission_files),
                "rubrics": len(bundle.rubric_files),
                "briefs": len(bundle.brief_files),
                "marking_schemes": len(bundle.marking_scheme_files),
                "graded_samples": len(bundle.graded_sample_files),
                "other_docs": len(bundle.other_files),
            }
        )
    return rows


def parse_keyword_setting(value: str, default_key: str) -> tuple[str, ...]:
    parsed = parse_keyword_patterns(value)
    return parsed or DEFAULT_CONTEXT_PATTERNS[default_key]


def build_exported_rubric_text(contexts_by_assessment: dict[str, MarkingContext]) -> str:
    sections: list[str] = []
    for assessment_name, context in contexts_by_assessment.items():
        rubric_text = build_rubric_matrix_markdown(context, assessment_name=assessment_name).strip()
        if not rubric_text:
            continue
        sections.append(rubric_text)
    return "\n\n".join(sections).strip()


def choose_directory(initial_dir: str = "") -> str:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(initialdir=initial_dir or None)
    finally:
        root.destroy()
    return selected


def read_csv_source(csv_source: object) -> pd.DataFrame:
    if isinstance(csv_source, Path):
        try:
            return pd.read_csv(csv_source)
        except UnicodeDecodeError:
            return pd.read_csv(csv_source, encoding="cp1252")

    csv_source.seek(0)
    try:
        return pd.read_csv(csv_source)
    except UnicodeDecodeError:
        csv_source.seek(0)
        return pd.read_csv(csv_source, encoding="cp1252")


def load_submission_source_text(submission_source: object) -> tuple[str, str]:
    if isinstance(submission_source, Path):
        script_text, _ = build_submission_texts_from_path(submission_source)
        return script_text, submission_source.name
    script_text, _ = build_submission_texts_from_upload(submission_source)
    return script_text, submission_source.name


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
    model_examples = ", ".join(model_name for model_name, _ in AVAILABLE_MODELS.values())
    st.markdown(
        "Choose the primary grader and, optionally, a verifier. Make sure Ollama is running and the models are already pulled "
        f"(e.g. `{model_examples}`)."
    )
    model_keys = list(AVAILABLE_MODELS.keys())
    grader_model_key = st.selectbox(
        "Primary grader",
        options=model_keys,
        index=model_keys.index("Qwen2 7B"),
    )
    use_verifier = st.checkbox("Use verifier", value=True)
    verifier_model_key = None
    if use_verifier:
        verifier_model_key = st.selectbox(
            "Verifier",
            options=model_keys,
            index=model_keys.index("Qwen2 7B"),
        )

    grader_model = AVAILABLE_MODELS[grader_model_key]
    selected_models = [grader_model]
    verifier_model = AVAILABLE_MODELS[verifier_model_key] if verifier_model_key else None

    st.header("Step 2 - Mark Scale")
    st.caption("The final mark is always the deterministic sum of the section marks. If there is only one section, that section mark is the overall mark.")
    scale_options = list(SCALE_PRESETS.keys())
    scale_profile = st.selectbox(
        "Choose the grading scale",
        options=scale_options,
        index=scale_options.index("Detect from documents"),
    )
    manual_max_mark = None
    if scale_profile == "Custom":
        manual_max_mark = st.number_input(
            "Custom maximum mark",
            min_value=1.0,
            step=1.0,
            value=100.0,
        )
    else:
        manual_max_mark = SCALE_PRESETS[scale_profile]
        if manual_max_mark is not None:
            st.info(f"Using a fixed {manual_max_mark:g}-point scale.")

    st.header("Step 3 - Student Submissions")
    use_assessment_folders = st.checkbox(
        "Load assessments from subfolders inside one parent folder",
        value=False,
        help="Use this when one parent folder contains one folder per assessment.",
    )
    single_assessment_folder_mode = False

    script_files = []
    csv_file = None
    rubric_text = ""
    brief_text = ""
    marking_scheme_text = ""
    graded_sample_text = ""
    other_context_text = ""
    assessment_bundles: list[AssessmentBundle] = []
    rubric_keywords = DEFAULT_CONTEXT_PATTERNS["rubric"]
    brief_keywords = DEFAULT_CONTEXT_PATTERNS["brief"]
    marking_scheme_keywords = DEFAULT_CONTEXT_PATTERNS["marking_scheme"]
    graded_sample_keywords = DEFAULT_CONTEXT_PATTERNS["graded_sample"]
    other_keywords = DEFAULT_CONTEXT_PATTERNS["other"]

    if use_assessment_folders:
        st.caption("Choose one parent folder. Each child folder will be treated as a separate assessment.")
        single_assessment_folder_mode = st.checkbox(
            "Selected folder is one assessment with many student subfolders",
            value=False,
            help="Use this when the chosen folder contains student folders for a single assessment rather than multiple assessment folders.",
        )

        browse_col, path_col = st.columns([1, 4])
        with browse_col:
            if st.button("Browse folders", use_container_width=True):
                selected_dir = choose_directory(st.session_state.get("assessment_root", ""))
                if selected_dir:
                    st.session_state["assessment_root"] = selected_dir
        with path_col:
            st.text_input(
                "Selected parent folder",
                value=st.session_state.get("assessment_root", ""),
                key="assessment_root_display",
                disabled=True,
            )

        assessment_root = st.session_state.get("assessment_root", "").strip()

        with st.expander("Folder matching rules"):
            rubric_keywords = parse_keyword_setting(
                st.text_input("Rubric filename keywords", value=", ".join(DEFAULT_CONTEXT_PATTERNS["rubric"])),
                "rubric",
            )
            brief_keywords = parse_keyword_setting(
                st.text_input("Brief filename keywords", value=", ".join(DEFAULT_CONTEXT_PATTERNS["brief"])),
                "brief",
            )
            marking_scheme_keywords = parse_keyword_setting(
                st.text_input("Marking scheme filename keywords", value=", ".join(DEFAULT_CONTEXT_PATTERNS["marking_scheme"])),
                "marking_scheme",
            )
            graded_sample_keywords = parse_keyword_setting(
                st.text_input("Example graded script keywords", value=", ".join(DEFAULT_CONTEXT_PATTERNS["graded_sample"])),
                "graded_sample",
            )
            other_keywords = parse_keyword_setting(
                st.text_input("Other supporting document keywords", value=", ".join(DEFAULT_CONTEXT_PATTERNS["other"])),
                "other",
            )

        if assessment_root:
            try:
                if single_assessment_folder_mode:
                    assessment_bundles = [
                        build_single_assessment_bundle(
                            Path(assessment_root),
                            assessment_name=Path(assessment_root).name,
                            rubric_keywords=rubric_keywords,
                            brief_keywords=brief_keywords,
                            marking_scheme_keywords=marking_scheme_keywords,
                            graded_sample_keywords=graded_sample_keywords,
                            other_keywords=other_keywords,
                        )
                    ]
                else:
                    assessment_bundles = discover_assessment_bundles(
                        Path(assessment_root),
                        rubric_keywords=rubric_keywords,
                        brief_keywords=brief_keywords,
                        marking_scheme_keywords=marking_scheme_keywords,
                        graded_sample_keywords=graded_sample_keywords,
                        other_keywords=other_keywords,
                    )
            except ValueError as exc:
                st.error(str(exc))
            else:
                if assessment_bundles:
                    st.caption("Discovered assessment folders")
                    st.dataframe(pd.DataFrame(build_bundle_summary_rows(assessment_bundles)), use_container_width=True)
                else:
                    st.warning("No assessment folders were found under the selected parent folder.")
    else:
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

    st.header("Step 4 - Marking Documents")
    if use_assessment_folders:
        st.caption(
            "Upload shared marking documents here when they should apply to every discovered assessment folder. "
            "Folder-specific documents will still be used when present."
        )
    else:
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

    scale_suggestion = suggest_marking_scale(
        rubric_text=rubric_text,
        brief_text=brief_text,
        marking_scheme_text=marking_scheme_text,
        graded_sample_text=graded_sample_text,
        other_context_text=other_context_text,
    )
    if scale_suggestion["max_mark"] is not None:
        st.caption(f"Document suggestion: {scale_suggestion['label']} ({scale_suggestion['reason']})")
    else:
        st.caption(f"Document suggestion: {scale_suggestion['label']} ({scale_suggestion['reason']})")

    if manual_max_mark is not None and scale_suggestion["max_mark"] is not None and abs(float(manual_max_mark) - float(scale_suggestion["max_mark"])) > 1e-9:
        st.warning(
            f"Chosen scale is {manual_max_mark:g}, but the uploaded documents suggest {float(scale_suggestion['max_mark']):g}. "
            "The chosen scale will be used."
        )

    st.header("Step 5 - Warm Test Snapshot")
    last_snapshot = load_ingest_snapshot()
    if last_snapshot is not None:
        st.caption(f"Saved snapshot: `{get_last_ingest_manifest_path()}`")
    use_saved_snapshot = st.checkbox(
        "Use saved warm-test snapshot for this run",
        value=False,
        disabled=last_snapshot is None,
        help="Reuse the most recently saved ingest without re-uploading files.",
    )
    if st.button("Save current ingest snapshot"):
        try:
            snapshot_path = save_ingest_snapshot(
                use_assessment_folders=use_assessment_folders,
                single_assessment_folder_mode=single_assessment_folder_mode,
                assessment_root=st.session_state.get("assessment_root", ""),
                folder_keywords={
                    "rubric": rubric_keywords,
                    "brief": brief_keywords,
                    "marking_scheme": marking_scheme_keywords,
                    "graded_sample": graded_sample_keywords,
                    "other": other_keywords,
                },
                scale_profile=scale_profile,
                manual_max_mark=manual_max_mark,
                document_only_mode=st.session_state.get("document_only_mode", False),
                script_files=script_files,
                csv_file=csv_file,
                rubric_files=rubric_files,
                brief_files=brief_files,
                marking_scheme_files=marking_scheme_files,
                graded_sample_files=graded_sample_files,
                other_files=other_files,
            )
        except Exception as exc:
            st.error(f"Could not save ingest snapshot: {exc}")
        else:
            st.success(f"Saved warm-test snapshot to {snapshot_path}")

    run_step_label = "Step 6 - Run Marking"
    st.header(run_step_label)
    if st.button("Run marking", type="primary"):
        if not selected_models:
            st.error("Select at least one model before running the marking.")
            return

        run_use_assessment_folders = use_assessment_folders
        run_single_assessment_folder_mode = single_assessment_folder_mode
        run_assessment_bundles = assessment_bundles
        run_script_files: list[object] = list(script_files)
        run_csv_file: object | None = csv_file
        run_rubric_text = rubric_text
        run_brief_text = brief_text
        run_marking_scheme_text = marking_scheme_text
        run_graded_sample_text = graded_sample_text
        run_other_context_text = other_context_text
        run_manual_max_mark = manual_max_mark
        run_document_only_mode = st.session_state.get("document_only_mode", False)

        if use_saved_snapshot:
            snapshot = load_ingest_snapshot()
            if snapshot is None:
                st.error("No saved warm-test snapshot is available.")
                return
            run_use_assessment_folders = bool(snapshot.get("use_assessment_folders"))
            run_single_assessment_folder_mode = bool(snapshot.get("single_assessment_folder_mode"))
            run_manual_max_mark = snapshot.get("manual_max_mark")
            run_document_only_mode = bool(snapshot.get("document_only_mode"))
            documents = snapshot.get("documents", {})
            run_rubric_text = read_paths_text([Path(path) for path in documents.get("rubric_files", [])])
            run_brief_text = read_paths_text([Path(path) for path in documents.get("brief_files", [])])
            run_marking_scheme_text = read_paths_text([Path(path) for path in documents.get("marking_scheme_files", [])])
            run_graded_sample_text = read_paths_text([Path(path) for path in documents.get("graded_sample_files", [])])
            run_other_context_text = read_paths_text([Path(path) for path in documents.get("other_files", [])])

            if run_use_assessment_folders:
                assessment_root = str(snapshot.get("assessment_root", "")).strip()
                if not assessment_root:
                    st.error("Saved snapshot does not contain a valid assessment folder path.")
                    return
                folder_keywords = snapshot.get("folder_keywords", {})
                snapshot_rubric_keywords = tuple(folder_keywords.get("rubric", DEFAULT_CONTEXT_PATTERNS["rubric"]))
                snapshot_brief_keywords = tuple(folder_keywords.get("brief", DEFAULT_CONTEXT_PATTERNS["brief"]))
                snapshot_marking_scheme_keywords = tuple(folder_keywords.get("marking_scheme", DEFAULT_CONTEXT_PATTERNS["marking_scheme"]))
                snapshot_graded_sample_keywords = tuple(folder_keywords.get("graded_sample", DEFAULT_CONTEXT_PATTERNS["graded_sample"]))
                snapshot_other_keywords = tuple(folder_keywords.get("other", DEFAULT_CONTEXT_PATTERNS["other"]))
                try:
                    if run_single_assessment_folder_mode:
                        run_assessment_bundles = [
                            build_single_assessment_bundle(
                                Path(assessment_root),
                                assessment_name=Path(assessment_root).name,
                                rubric_keywords=snapshot_rubric_keywords,
                                brief_keywords=snapshot_brief_keywords,
                                marking_scheme_keywords=snapshot_marking_scheme_keywords,
                                graded_sample_keywords=snapshot_graded_sample_keywords,
                                other_keywords=snapshot_other_keywords,
                            )
                        ]
                    else:
                        run_assessment_bundles = discover_assessment_bundles(
                            Path(assessment_root),
                            rubric_keywords=snapshot_rubric_keywords,
                            brief_keywords=snapshot_brief_keywords,
                            marking_scheme_keywords=snapshot_marking_scheme_keywords,
                            graded_sample_keywords=snapshot_graded_sample_keywords,
                            other_keywords=snapshot_other_keywords,
                        )
                except ValueError as exc:
                    st.error(f"Saved snapshot could not be loaded: {exc}")
                    return
            else:
                run_script_files = [Path(path) for path in snapshot.get("script_files", [])]
                run_csv_file = Path(snapshot["csv_file"]) if snapshot.get("csv_file") else None

        rows: list[dict[str, object]] = []
        progress = st.progress(0)
        status = st.empty()
        exported_rubric_text = ""

        if run_use_assessment_folders:
            if not run_assessment_bundles:
                st.error("Enter a valid parent folder that contains one or more assessment subfolders.")
                return

            total = sum(len(bundle.submission_files) for bundle in run_assessment_bundles)
            if total == 0:
                st.error("No student submissions were found in the discovered assessment folders.")
                return

            processed = 0
            max_marks: dict[str, float] = {}
            contexts_by_assessment: dict[str, MarkingContext] = {}

            for bundle in run_assessment_bundles:
                try:
                    rubric_bundle_text = combine_text_sections(read_paths_text(bundle.rubric_files), run_rubric_text)
                    brief_bundle_text = combine_text_sections(read_paths_text(bundle.brief_files), run_brief_text)
                    marking_scheme_bundle_text = combine_text_sections(
                        read_paths_text(bundle.marking_scheme_files),
                        run_marking_scheme_text,
                    )
                    graded_sample_bundle_text = combine_text_sections(
                        read_paths_text(bundle.graded_sample_files),
                        run_graded_sample_text,
                    )
                    other_bundle_text = combine_text_sections(read_paths_text(bundle.other_files), run_other_context_text)

                    effective_rubric_text = rubric_bundle_text
                    if not effective_rubric_text.strip() and run_document_only_mode:
                        effective_rubric_text = build_document_based_marking_text(
                            brief_text=brief_bundle_text,
                            marking_scheme_text=marking_scheme_bundle_text,
                            graded_sample_text=graded_sample_bundle_text,
                            other_context_text=other_bundle_text,
                        )

                    marking_context = build_marking_context_with_optional_override(
                        rubric_text=effective_rubric_text,
                        brief_text=brief_bundle_text,
                        marking_scheme_text=marking_scheme_bundle_text,
                        graded_sample_text=graded_sample_bundle_text,
                        other_context_text=other_bundle_text,
                        manual_max_mark=run_manual_max_mark,
                    )
                except ValueError as exc:
                    st.error(f"{bundle.name}: {exc}")
                    return

                contexts_by_assessment[bundle.name] = marking_context
                max_marks[bundle.name] = marking_context.max_mark
                sidebar_status.info(f"Marking {bundle.name} ({marking_context.max_mark:g}-point scale)")

                for submission_path in bundle.submission_files:
                    processed += 1
                    status.text(f"Marking {bundle.name} / {submission_path.name} ({processed}/{total})")

                    row: dict[str, object] = {
                        "assessment_name": bundle.name,
                        "filename": submission_path.name,
                        "source_path": str(submission_path),
                    }
                    try:
                        script_text, structure_script_text = build_submission_texts_from_path(submission_path)
                    except Exception as exc:
                        for _, label in selected_models:
                            apply_model_result(row, label, error=exc)
                        if verifier_model is not None:
                            apply_verifier_result(row, verifier_model[1], error=exc)
                        rows.append(row)
                        progress.progress(processed / total)
                        continue

                    for model_name, label in selected_models:
                        try:
                            result = call_ollama(
                                script_text=script_text,
                                structure_script_text=structure_script_text,
                                context=marking_context,
                                filename=submission_path.name,
                                model_name=model_name,
                                rubric_verifier_model_name=verifier_model[0] if verifier_model is not None else None,
                            )
                            apply_model_result(row, label, result=result)
                        except Exception as exc:
                            apply_model_result(row, label, error=exc)

                    maybe_run_regrade_loop(
                        row=row,
                        script_text=script_text,
                        context=marking_context,
                        filename=submission_path.name,
                        grader_model=grader_model,
                        verifier_model=verifier_model,
                    )

                    rows.append(row)
                    progress.progress(processed / total)

            df = pd.DataFrame(rows)
            df = apply_cross_student_calibration(df, selected_models, contexts_by_assessment)
            report_sections = []
            for assessment_name, assessment_df in df.groupby("assessment_name", sort=True):
                report_sections.append(
                    build_consistency_report(
                        assessment_df.reset_index(drop=True),
                        selected_models,
                        max_marks[assessment_name],
                    ).strip()
                )
            consistency_report = "\n\n".join(report_sections) + "\n"
            exported_rubric_text = build_exported_rubric_text(contexts_by_assessment)
        elif run_csv_file is not None:
            try:
                effective_rubric_text = run_rubric_text
                if not effective_rubric_text.strip() and run_document_only_mode:
                    effective_rubric_text = build_document_based_marking_text(
                        brief_text=run_brief_text,
                        marking_scheme_text=run_marking_scheme_text,
                        graded_sample_text=run_graded_sample_text,
                        other_context_text=run_other_context_text,
                    )

                marking_context = build_marking_context_with_optional_override(
                    rubric_text=effective_rubric_text,
                    brief_text=run_brief_text,
                    marking_scheme_text=run_marking_scheme_text,
                    graded_sample_text=run_graded_sample_text,
                    other_context_text=run_other_context_text,
                    manual_max_mark=run_manual_max_mark,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

            sidebar_status.info(f"Marking in progress ({marking_context.max_mark:g}-point scale)")
            try:
                df_input = read_csv_source(run_csv_file)
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
                            rubric_verifier_model_name=verifier_model[0] if verifier_model is not None else None,
                        )
                        apply_model_result(row, label, result=result)
                    except Exception as exc:
                        apply_model_result(row, label, error=exc)

                maybe_run_regrade_loop(
                    row=row,
                    script_text=script_text,
                    context=marking_context,
                    filename=filename,
                    grader_model=grader_model,
                    verifier_model=verifier_model,
                )

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
            df = apply_cross_student_calibration(df, selected_models, {"All Submissions": marking_context})
            consistency_report = build_consistency_report(df, selected_models, marking_context.max_mark)
            exported_rubric_text = marking_context.rubric_text
        else:
            if not run_script_files:
                st.error("Upload at least one script or a CSV file.")
                return

            try:
                effective_rubric_text = run_rubric_text
                if not effective_rubric_text.strip() and run_document_only_mode:
                    effective_rubric_text = build_document_based_marking_text(
                        brief_text=run_brief_text,
                        marking_scheme_text=run_marking_scheme_text,
                        graded_sample_text=run_graded_sample_text,
                        other_context_text=run_other_context_text,
                    )

                marking_context = build_marking_context_with_optional_override(
                    rubric_text=effective_rubric_text,
                    brief_text=run_brief_text,
                    marking_scheme_text=run_marking_scheme_text,
                    graded_sample_text=run_graded_sample_text,
                    other_context_text=run_other_context_text,
                    manual_max_mark=run_manual_max_mark,
                )
            except ValueError as exc:
                st.error(str(exc))
                return

            sidebar_status.info(f"Marking in progress ({marking_context.max_mark:g}-point scale)")
            total = len(run_script_files)
            for index, uploaded in enumerate(run_script_files, start=1):
                script_text, structure_script_text = build_submission_texts_from_upload(uploaded)
                filename = uploaded.name
                status.text(f"Marking {filename} ({index}/{total})")

                row = {"filename": filename}
                for model_name, label in selected_models:
                    try:
                        result = call_ollama(
                            script_text=script_text,
                            structure_script_text=structure_script_text,
                            context=marking_context,
                            filename=filename,
                            model_name=model_name,
                            rubric_verifier_model_name=verifier_model[0] if verifier_model is not None else None,
                        )
                        apply_model_result(row, label, result=result)
                    except Exception as exc:
                        apply_model_result(row, label, error=exc)

                maybe_run_regrade_loop(
                    row=row,
                    script_text=script_text,
                    context=marking_context,
                    filename=filename,
                    grader_model=grader_model,
                    verifier_model=verifier_model,
                )

                rows.append(row)
                progress.progress(index / total)

            df = pd.DataFrame(rows)
            df = apply_cross_student_calibration(df, selected_models, {"All Submissions": marking_context})
            consistency_report = build_consistency_report(df, selected_models, marking_context.max_mark)
            exported_rubric_text = marking_context.rubric_text

        if df.empty:
            st.warning("No results to show.")
            sidebar_status.info("Ready")
            return

        if verifier_model is not None:
            verifier_report = build_verifier_report(df, verifier_model[1])
            if verifier_report.strip():
                consistency_report = consistency_report.rstrip() + "\n\n" + verifier_report

        mark_columns = [column for column in df.columns if column.endswith("_mark")]
        if len(mark_columns) >= 2:
            df["average_mark"] = df[mark_columns].mean(axis=1)
            if len(mark_columns) == 2:
                df["mark_difference"] = (df[mark_columns[0]] - df[mark_columns[1]]).abs()

        st.success("Marking completed.")
        sidebar_status.success("Completed")
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
        render_results_bundle_download(df, consistency_report, rubric_text=exported_rubric_text)
        render_consistency_report(consistency_report)
        render_rubric_matrix(exported_rubric_text)


if __name__ == "__main__":
    main()
