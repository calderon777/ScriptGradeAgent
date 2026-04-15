from typing import Any

import pandas as pd

from .core import (
    MarkingContext,
    calibrate_marks_across_students,
    infer_max_mark_from_texts,
    prepare_marking_context,
    regrade_marking_result,
    verify_marking_result,
)

SCALE_PRESETS: dict[str, float | None] = {
    "Detect from documents": None,
    "Deterministic / numeric (0-100)": 100.0,
    "Analytical / evaluative (0-85, typical 20-80)": 85.0,
    "Custom": None,
}


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
    if manual_max_mark is not None:
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


def suggest_marking_scale(
    rubric_text: str,
    brief_text: str,
    marking_scheme_text: str,
    graded_sample_text: str,
    other_context_text: str,
) -> dict[str, object]:
    try:
        inferred = infer_max_mark_from_texts(
            rubric_text,
            marking_scheme_text,
            brief_text,
            graded_sample_text,
            other_context_text,
        )
    except ValueError as exc:
        return {
            "label": "Conflicting document scales",
            "max_mark": None,
            "reason": str(exc),
        }

    if inferred is None:
        return {
            "label": "No document suggestion",
            "max_mark": None,
            "reason": "The uploaded documents do not state a clear overall mark.",
        }
    if abs(inferred - 100.0) < 1e-9:
        label = "Deterministic / numeric (0-100)"
    elif abs(inferred - 85.0) < 1e-9:
        label = "Analytical / evaluative (0-85, typical 20-80)"
    else:
        label = f"Inferred {inferred:g}-point scale"
    return {
        "label": label,
        "max_mark": float(inferred),
        "reason": f"Documents suggest an overall maximum mark of {inferred:g}.",
    }


def combine_text_sections(*texts: str) -> str:
    return "\n\n".join(text.strip() for text in texts if text and text.strip()).strip()


def apply_model_result(row: dict[str, object], label: str, result: dict[str, object] | None = None, error: Exception | None = None) -> None:
    row[f"{label}_status"] = "ok" if error is None else "error"
    row[f"{label}_error"] = "" if error is None else str(error)
    if result is None:
        row[f"{label}_mark"] = None
        row[f"{label}_ai_total_mark"] = None
        row[f"{label}_math_total_mark"] = None
        row[f"{label}_ai_math_mark_delta"] = None
        row[f"{label}_latency_seconds_total"] = None
        row[f"{label}_latency_seconds_assessment_prepare"] = None
        row[f"{label}_latency_seconds_rubric_verify"] = None
        row[f"{label}_latency_seconds_structure_detect"] = None
        row[f"{label}_latency_seconds_part_refine"] = None
        row[f"{label}_latency_seconds_part_analysis"] = None
        row[f"{label}_latency_seconds_part_analysis_per_part_avg"] = None
        row[f"{label}_latency_seconds_moderation"] = None
        row[f"{label}_latency_seconds_finalize"] = None
        row[f"{label}_feedback"] = ""
        row[f"{label}_strengths"] = ""
        row[f"{label}_weaknesses"] = ""
        row[f"{label}_detected_part_count"] = None
        row[f"{label}_detected_part_labels"] = ""
        row[f"{label}_extracted_word_count"] = None
        row[f"{label}_possible_extraction_issue"] = None
        row[f"{label}_validation_notes"] = ""
        row[f"{label}_calibration_delta"] = None
        row[f"{label}_calibration_rationale"] = ""
        return
    row[f"{label}_mark"] = result["total_mark"]
    row[f"{label}_ai_total_mark"] = result.get("ai_total_mark")
    row[f"{label}_math_total_mark"] = result.get("math_total_mark")
    row[f"{label}_ai_math_mark_delta"] = result.get("ai_math_mark_delta")
    row[f"{label}_latency_seconds_total"] = result.get("latency_seconds_total")
    row[f"{label}_latency_seconds_assessment_prepare"] = result.get("latency_seconds_assessment_prepare")
    row[f"{label}_latency_seconds_rubric_verify"] = result.get("latency_seconds_rubric_verify")
    row[f"{label}_latency_seconds_structure_detect"] = result.get("latency_seconds_structure_detect")
    row[f"{label}_latency_seconds_part_refine"] = result.get("latency_seconds_part_refine")
    row[f"{label}_latency_seconds_part_analysis"] = result.get("latency_seconds_part_analysis")
    row[f"{label}_latency_seconds_part_analysis_per_part_avg"] = result.get("latency_seconds_part_analysis_per_part_avg")
    row[f"{label}_latency_seconds_moderation"] = result.get("latency_seconds_moderation")
    row[f"{label}_latency_seconds_finalize"] = result.get("latency_seconds_finalize")
    row[f"{label}_feedback"] = result["overall_feedback"]
    row[f"{label}_strengths"] = " | ".join(result.get("strengths", []))
    row[f"{label}_weaknesses"] = " | ".join(result.get("weaknesses", []))
    row[f"{label}_detected_part_count"] = result.get("detected_part_count")
    row[f"{label}_detected_part_labels"] = " | ".join(result.get("detected_part_labels", []))
    row[f"{label}_extracted_word_count"] = result.get("extracted_word_count")
    row[f"{label}_possible_extraction_issue"] = result.get("possible_extraction_issue")
    row[f"{label}_validation_notes"] = " | ".join(str(item) for item in result.get("validation_notes", []))
    row[f"{label}_calibration_delta"] = None
    row[f"{label}_calibration_rationale"] = ""


def apply_verifier_result(row: dict[str, object], label: str, result: dict[str, object] | None = None, error: Exception | None = None) -> None:
    row[f"{label}_status"] = "ok" if error is None else "error"
    row[f"{label}_error"] = "" if error is None else str(error)
    if result is None:
        row[f"{label}_agreement"] = ""
        row[f"{label}_confidence"] = None
        row[f"{label}_issues"] = ""
        row[f"{label}_recommendation"] = ""
        return
    row[f"{label}_agreement"] = result["agreement"]
    row[f"{label}_confidence"] = result["confidence_0_to_100"]
    row[f"{label}_issues"] = " | ".join(result.get("issues", []))
    row[f"{label}_recommendation"] = result["recommendation"]


def maybe_run_regrade_loop(
    row: dict[str, object],
    script_text: str,
    context: MarkingContext,
    filename: str,
    grader_model: tuple[str, str],
    verifier_model: tuple[str, str] | None,
) -> None:
    if verifier_model is None:
        return

    grader_label = grader_model[1]
    verifier_label = verifier_model[1]
    if row.get(f"{grader_label}_status") != "ok":
        return

    try:
        verification = verify_marking_result(
            script_text=script_text,
            context=context,
            filename=filename,
            grading_result={
                "total_mark": row.get(f"{grader_label}_mark"),
                "overall_feedback": row.get(f"{grader_label}_feedback"),
                "strengths": [item.strip() for item in str(row.get(f"{grader_label}_strengths", "")).split("|") if item.strip()],
                "weaknesses": [item.strip() for item in str(row.get(f"{grader_label}_weaknesses", "")).split("|") if item.strip()],
                "covered_parts": [item.strip() for item in str(row.get(f"{grader_label}_detected_part_labels", "")).split("|") if item.strip()],
                "max_mark": context.max_mark,
            },
            model_name=verifier_model[0],
        )
        apply_verifier_result(row, verifier_label, verification)
    except Exception as exc:
        apply_verifier_result(row, verifier_label, error=exc)
        return

    if verification["agreement"] != "major_concern":
        return

    try:
        revised = regrade_marking_result(
            script_text=script_text,
            context=context,
            filename=filename,
            prior_result={
                "total_mark": row.get(f"{grader_label}_mark"),
                "max_mark": context.max_mark,
                "overall_feedback": row.get(f"{grader_label}_feedback"),
                "strengths": [item.strip() for item in str(row.get(f"{grader_label}_strengths", "")).split("|") if item.strip()],
                "weaknesses": [item.strip() for item in str(row.get(f"{grader_label}_weaknesses", "")).split("|") if item.strip()],
                "covered_parts": [item.strip() for item in str(row.get(f"{grader_label}_detected_part_labels", "")).split("|") if item.strip()],
            },
            verifier_result=verification,
            model_name=grader_model[0],
        )
        apply_model_result(row, grader_label, revised)
        existing = str(row.get(f"{grader_label}_validation_notes", "")).strip()
        row[f"{grader_label}_validation_notes"] = f"{existing} | regraded_after_verifier".strip(" |")
    except Exception as exc:
        existing = str(row.get(f"{grader_label}_validation_notes", "")).strip()
        row[f"{grader_label}_validation_notes"] = f"{existing} | regrade_failed: {exc}".strip(" |")
        return

    try:
        second_verification = verify_marking_result(
            script_text=script_text,
            context=context,
            filename=filename,
            grading_result={
                "total_mark": row.get(f"{grader_label}_mark"),
                "overall_feedback": row.get(f"{grader_label}_feedback"),
                "strengths": [item.strip() for item in str(row.get(f"{grader_label}_strengths", "")).split("|") if item.strip()],
                "weaknesses": [item.strip() for item in str(row.get(f"{grader_label}_weaknesses", "")).split("|") if item.strip()],
                "covered_parts": [item.strip() for item in str(row.get(f"{grader_label}_detected_part_labels", "")).split("|") if item.strip()],
                "max_mark": context.max_mark,
            },
            model_name=verifier_model[0],
        )
        apply_verifier_result(row, verifier_label, second_verification)
        if second_verification["agreement"] == "major_concern":
            existing = str(row.get(f"{grader_label}_validation_notes", "")).strip()
            row[f"{grader_label}_validation_notes"] = f"{existing} | review_required".strip(" |")
    except Exception as exc:
        apply_verifier_result(row, verifier_label, error=exc)


def apply_cross_student_calibration(
    df: pd.DataFrame,
    selected_models: list[tuple[str, str]],
    context_by_group: dict[str, MarkingContext],
) -> pd.DataFrame:
    calibrated_df = df.copy()
    if "assessment_name" in calibrated_df.columns:
        groups = [(name, group.index.tolist()) for name, group in calibrated_df.groupby("assessment_name", sort=False)]
    else:
        groups = [("All Submissions", calibrated_df.index.tolist())]

    for assessment_name, row_indexes in groups:
        context = context_by_group.get(assessment_name)
        if context is None:
            continue
        for model_name, label in selected_models:
            rows = []
            for row_index in row_indexes:
                row = calibrated_df.loc[row_index]
                if row.get(f"{label}_status") != "ok":
                    continue
                strengths = [item.strip() for item in str(row.get(f"{label}_strengths", "")).split("|") if item.strip()]
                weaknesses = [item.strip() for item in str(row.get(f"{label}_weaknesses", "")).split("|") if item.strip()]
                notes = [item.strip() for item in str(row.get(f"{label}_validation_notes", "")).split("|") if item.strip()]
                rows.append(
                    {
                        "filename": row["filename"],
                        "total_mark": row.get(f"{label}_mark"),
                        "detected_part_count": row.get(f"{label}_detected_part_count"),
                        "strengths": strengths,
                        "weaknesses": weaknesses,
                        "validation_notes": notes,
                    }
                )

            if len(rows) < 2:
                continue

            try:
                calibration = calibrate_marks_across_students(
                    provisional_results=rows,
                    model_name=model_name,
                    model_label=label,
                    context=context,
                    assessment_name=assessment_name,
                )
            except Exception as exc:
                for row_index in row_indexes:
                    existing = str(calibrated_df.at[row_index, f"{label}_validation_notes"] or "").strip()
                    note = f"calibration_failed: {exc}"
                    calibrated_df.at[row_index, f"{label}_validation_notes"] = f"{existing} | {note}".strip(" |")
                continue

            for row_index in row_indexes:
                filename = calibrated_df.at[row_index, "filename"]
                if filename not in calibration:
                    continue
                result = calibration[filename]
                calibrated_df.at[row_index, f"{label}_mark"] = result["adjusted_mark"]
                calibrated_df.at[row_index, f"{label}_calibration_delta"] = result["delta"]
                calibrated_df.at[row_index, f"{label}_calibration_rationale"] = result["rationale"]
                existing = str(calibrated_df.at[row_index, f"{label}_validation_notes"] or "").strip()
                calibrated_df.at[row_index, f"{label}_validation_notes"] = f"{existing} | calibrated".strip(" |")

    return calibrated_df
