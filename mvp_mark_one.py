from marking_pipeline import (
    ROOT_DIR,
    build_submission_texts_from_path,
    call_ollama,
    list_submission_files,
    prepare_marking_context,
)


SCRIPTS_DIR = ROOT_DIR / "scripts" / "test"
RUBRIC_FILE = ROOT_DIR / "rubrics" / "labour_year2_rubric.txt"

MODELS = [
    ("qwen2:7b", "Qwen2_7B"),
    ("llama3.1:8b", "Llama3.1_8B"),
]


def main() -> None:
    if not SCRIPTS_DIR.exists():
        print(f"Submission directory not found: {SCRIPTS_DIR}")
        return
    if not RUBRIC_FILE.exists():
        print(f"Rubric file not found: {RUBRIC_FILE}")
        return

    submission_paths = list_submission_files(SCRIPTS_DIR)
    if not submission_paths:
        print(f"No supported submission files found in {SCRIPTS_DIR}")
        return

    rubric_text = RUBRIC_FILE.read_text(encoding="utf-8")
    try:
        marking_context = prepare_marking_context(
            rubric_text=rubric_text,
            brief_text="",
            marking_scheme_text="",
            graded_sample_text="",
            other_context_text="",
        )
    except ValueError as exc:
        print(f"Cannot start marking: {exc}")
        return

    submission_path = submission_paths[0]
    try:
        script_text, structure_script_text = build_submission_texts_from_path(submission_path)
    except Exception as exc:
        print(f"Could not read {submission_path.name}: {exc}")
        return

    print(f"\n=== MARKING {submission_path.name} WITH MULTIPLE MODELS ===")
    for model_name, label in MODELS:
        try:
            result = call_ollama(
                script_text=script_text,
                structure_script_text=structure_script_text,
                context=marking_context,
                filename=submission_path.name,
                model_name=model_name,
            )
            print(f"\n--- {label} ---")
            print(f"Mark: {result['total_mark']}/{result['max_mark']}")
            print("\nFeedback:")
            print(result["overall_feedback"])
            print("------------------------")
        except Exception as exc:
            print(f"\n--- {label} ---")
            print(f"ERROR while using model {model_name}: {exc}")
            print("------------------------")


if __name__ == "__main__":
    main()
