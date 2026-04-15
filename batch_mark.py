from datetime import datetime

import pandas as pd

from marking_pipeline import (
    MarkingContext,
    ROOT_DIR,
    build_submission_texts_from_path,
    call_ollama,
    list_submission_files,
    prepare_marking_context,
)


SCRIPTS_DIR = ROOT_DIR / "scripts" / "test"
RUBRIC_FILE = ROOT_DIR / "rubrics" / "labour_year2_rubric.txt"
OUTPUT_DIR = ROOT_DIR / "output"

MODELS = [
    ("qwen2:7b", "Qwen2_7B"),
    ("llama3.1:8b", "Llama3.1_8B"),
]


def load_marking_context() -> MarkingContext:
    rubric_text = RUBRIC_FILE.read_text(encoding="utf-8")
    return prepare_marking_context(
        rubric_text=rubric_text,
        brief_text="",
        marking_scheme_text="",
        graded_sample_text="",
        other_context_text="",
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


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not SCRIPTS_DIR.exists():
        print(f"Submission directory not found: {SCRIPTS_DIR}")
        return
    if not RUBRIC_FILE.exists():
        print(f"Rubric file not found: {RUBRIC_FILE}")
        return

    try:
        marking_context = load_marking_context()
    except ValueError as exc:
        print(f"Cannot start marking: {exc}")
        return

    submission_paths = list_submission_files(SCRIPTS_DIR)
    if not submission_paths:
        print(f"No supported submission files found in {SCRIPTS_DIR}")
        return

    rows: list[dict[str, object]] = []
    for submission_path in submission_paths:
        print(f"\nMarking: {submission_path.name}")
        row: dict[str, object] = {"filename": submission_path.name}
        try:
            script_text, structure_script_text = build_submission_texts_from_path(submission_path)
        except Exception as exc:
            print(f"  Could not read file: {exc}")
            continue

        for model_name, label in MODELS:
            try:
                result = call_ollama(
                    script_text=script_text,
                    structure_script_text=structure_script_text,
                    context=marking_context,
                    filename=submission_path.name,
                    model_name=model_name,
                )
                apply_model_result(row, label, result=result)
                print(f"  {label}: {result['total_mark']}/{result['max_mark']}")
            except Exception as exc:
                apply_model_result(row, label, error=exc)
                print(f"  {label}: ERROR - {exc}")
        rows.append(row)

    if not rows:
        print("No results were produced.")
        return

    df = pd.DataFrame(rows)
    if "Qwen2_7B_mark" in df.columns and "Llama3.1_8B_mark" in df.columns:
        df["average_mark"] = df[["Qwen2_7B_mark", "Llama3.1_8B_mark"]].mean(axis=1)
        df["mark_difference"] = (df["Qwen2_7B_mark"] - df["Llama3.1_8B_mark"]).abs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"marks_{timestamp}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nDone. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
