import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marking_pipeline.core import build_submission_texts_from_path, extract_message_content, parse_json_object  # noqa: E402
from scripts.run_human_benchmark import DEFAULT_OLLAMA_URL, DEFAULT_SAMPLE_PATHS, DEFAULT_WORKBOOK, load_context, load_human_records, participant_id_from_path  # noqa: E402


DEFAULT_MODEL = "qwen2:7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple one-shot experiment: send the answer and the solution to Qwen and ask it to assess like a field professor.")
    parser.add_argument("--sample", default=str(DEFAULT_SAMPLE_PATHS[0]), help="Submission path to assess.")
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK), help="Workbook path for human mark comparison.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama chat endpoint.")
    parser.add_argument("--num-ctx", type=int, default=24576, help="Context budget for the simple one-shot experiment.")
    parser.add_argument("--num-predict", type=int, default=700, help="Generation budget.")
    parser.add_argument(
        "--output",
        default=str(Path("output") / "simple_professor_assess_sadik_qwen.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def build_messages(marking_scheme_text: str, submission_text: str, max_mark: float) -> list[dict[str, str]]:
    system = "You are a professor of economics assessing a university coursework submission."
    user = (
        f"Maximum mark: {int(max_mark) if float(max_mark).is_integer() else max_mark}\n\n"
        "Official solution / marking material:\n"
        f"{marking_scheme_text.strip()}\n\n"
        "Student submission:\n"
        f"{submission_text.strip()}\n\n"
        "Assess the submission like a professor in this field. "
        "Return one JSON object with keys total_mark, max_mark, overall_feedback, strengths, weaknesses."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_ollama_json(
    model_name: str,
    messages: list[dict[str, str]],
    ollama_url: str,
    num_ctx: int,
    num_predict: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "format": "json",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.8,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "seed": 7,
        },
    }
    response = requests.post(ollama_url, json=payload, timeout=900)
    response.raise_for_status()
    content = extract_message_content(response.json())
    return parse_json_object(content)


def normalize_result(data: dict[str, Any], max_mark: float) -> dict[str, Any]:
    total_mark = data.get("total_mark")
    if isinstance(total_mark, bool) or not isinstance(total_mark, (int, float)):
        raise ValueError(f"Model did not return a numeric total_mark: {data}")
    total_value = float(total_mark)
    if total_value < 0 or total_value > max_mark:
        raise ValueError(f"Model returned an out-of-range total_mark: {data}")
    returned_max_mark = data.get("max_mark", max_mark)
    if isinstance(returned_max_mark, bool) or not isinstance(returned_max_mark, (int, float)):
        raise ValueError(f"Model did not return a numeric max_mark: {data}")
    feedback = str(data.get("overall_feedback", "")).strip()
    strengths = [str(item).strip() for item in data.get("strengths", []) if str(item).strip()] if isinstance(data.get("strengths", []), list) else []
    weaknesses = [str(item).strip() for item in data.get("weaknesses", []) if str(item).strip()] if isinstance(data.get("weaknesses", []), list) else []
    return {
        "total_mark": round(total_value, 2),
        "max_mark": float(returned_max_mark),
        "overall_feedback": feedback,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context = load_context()
    submission_text, _ = build_submission_texts_from_path(sample_path)
    messages = build_messages(
        marking_scheme_text=context.marking_scheme_text,
        submission_text=submission_text,
        max_mark=context.max_mark,
    )
    raw = call_ollama_json(
        model_name=args.model,
        messages=messages,
        ollama_url=args.ollama_url,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    normalized = normalize_result(raw, max_mark=context.max_mark)

    participant_id = participant_id_from_path(sample_path)
    human_records = load_human_records(Path(args.workbook))
    human_record = human_records.get(participant_id)
    payload = {
        "sample_path": str(sample_path),
        "model_name": args.model,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "prompt_words": sum(len(item["content"].split()) for item in messages),
        "submission_words": len(submission_text.split()),
        "marking_scheme_words": len((context.marking_scheme_text or "").split()),
        "human_total": human_record.total_mark if human_record else None,
        "human_feedback_excerpt": (re.sub(r"\s+", " ", human_record.feedback_comments).strip()[:220] + "...") if human_record and human_record.feedback_comments else "",
        "raw_result": raw,
        "normalized_result": normalized,
        "mark_delta": round(normalized["total_mark"] - human_record.total_mark, 2) if human_record else None,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
