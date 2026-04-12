import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
import requests
from docx import Document


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass(frozen=True)
class MarkingContext:
    rubric_text: str
    brief_text: str
    marking_scheme_text: str
    graded_sample_text: str
    other_context_text: str
    max_mark: float


def decode_text_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the text file with a supported encoding.")


def extract_text_from_pdf(file_obj: Any) -> str:
    text_parts: list[str] = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_obj: Any) -> str:
    doc = Document(file_obj)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()


def read_file_bytes(name: str, raw: bytes) -> str:
    lowered = name.lower()
    if lowered.endswith(".txt"):
        return decode_text_bytes(raw).strip()
    if lowered.endswith(".pdf"):
        return extract_text_from_pdf(io.BytesIO(raw))
    if lowered.endswith(".docx"):
        return extract_text_from_docx(io.BytesIO(raw))
    raise ValueError(f"Unsupported file type for {name}.")


def read_uploaded_files_text(uploaded_files: list[Any] | None) -> str:
    if not uploaded_files:
        return ""
    texts = [read_file_bytes(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files]
    return "\n\n".join(text for text in texts if text).strip()


def read_path_text(path: Path) -> str:
    return read_file_bytes(path.name, path.read_bytes())


def infer_max_mark_from_texts(*texts: str) -> float | None:
    patterns = (
        r"maximum\s+mark\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        r"max(?:imum)?\s+mark\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        r"single\s+overall\s+mark\s+out\s+of\s*(\d+(?:\.\d+)?)",
        r"mark\s+out\s+of\s*(\d+(?:\.\d+)?)",
        r"out\s+of\s*(\d+(?:\.\d+)?)",
        r"total\s+(?:mark|marks)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
    )

    candidates: set[float] = set()
    for text in texts:
        if not text:
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                candidates.add(float(match.group(1)))

    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Found conflicting maximum marks in the uploaded marking documents: "
            + ", ".join(_format_number(value) for value in sorted(candidates))
        )
    return next(iter(candidates))


def prepare_marking_context(
    rubric_text: str,
    brief_text: str,
    marking_scheme_text: str,
    graded_sample_text: str,
    other_context_text: str,
) -> MarkingContext:
    rubric_text = rubric_text.strip()
    brief_text = brief_text.strip()
    marking_scheme_text = marking_scheme_text.strip()
    graded_sample_text = graded_sample_text.strip()
    other_context_text = other_context_text.strip()

    if not (rubric_text or marking_scheme_text):
        raise ValueError("Provide a rubric or marking scheme before grading.")

    max_mark = infer_max_mark_from_texts(rubric_text, marking_scheme_text, brief_text)
    if max_mark is None:
        raise ValueError(
            "Could not infer the maximum mark from the rubric or marking scheme. "
            "Include an explicit phrase such as 'Maximum mark: 20' or 'out of 85'."
        )

    return MarkingContext(
        rubric_text=rubric_text,
        brief_text=brief_text,
        marking_scheme_text=marking_scheme_text,
        graded_sample_text=graded_sample_text,
        other_context_text=other_context_text,
        max_mark=max_mark,
    )


def build_messages(script_text: str, context: MarkingContext, filename: str) -> list[dict[str, str]]:
    max_mark_text = _format_number(context.max_mark)

    context_parts = []
    if context.rubric_text:
        context_parts.append(f"RUBRIC:\n{context.rubric_text}")
    if context.brief_text:
        context_parts.append(f"ASSIGNMENT BRIEF:\n{context.brief_text}")
    if context.marking_scheme_text:
        context_parts.append(f"MARKING SCHEME:\n{context.marking_scheme_text}")
    if context.graded_sample_text:
        context_parts.append(
            "EXAMPLE GRADED SCRIPT (for tone and structure only):\n"
            f"{context.graded_sample_text}"
        )
    if context.other_context_text:
        context_parts.append(f"OTHER SUPPORTING DOCUMENTS:\n{context.other_context_text}")

    system = (
        "You are a fair and consistent university examiner. "
        "Use only the supplied marking documents and the student's submission. "
        "Do not invent missing criteria or assume a different marking scale."
    )

    user = (
        "You are marking a student's script.\n\n"
        f"{'\n\n'.join(context_parts)}\n\n"
        f"STUDENT FILE NAME: {filename}\n\n"
        "STUDENT ANSWER:\n"
        f"\"\"\"{script_text}\"\"\"\n\n"
        "Return only one JSON object with exactly these keys:\n"
        '- "total_mark": number\n'
        '- "max_mark": number\n'
        '- "overall_feedback": string\n\n'
        f"Rules:\n"
        f"1. Use max_mark = {max_mark_text}.\n"
        f"2. total_mark must be between 0 and {max_mark_text}.\n"
        "3. overall_feedback must be at least 120 words.\n"
        "4. overall_feedback must include at least two concrete strengths and two concrete weaknesses.\n"
        "5. If the assessment has multiple questions, explicitly mention each question.\n"
        "6. Do not include markdown fences or any text outside the JSON object."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_ollama(
    script_text: str,
    context: MarkingContext,
    filename: str,
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    script_text = script_text.strip()
    if not script_text:
        raise ValueError(
            "No text could be extracted from the submission. "
            "OCR is not implemented yet, so use a text-based PDF, DOCX, or CSV answer."
        )

    payload = {
        "model": model_name,
        "format": "json",
        "messages": build_messages(script_text, context, filename),
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }

    response = requests.post(ollama_url, json=payload, timeout=300)
    response.raise_for_status()
    response_json = response.json()
    content = extract_message_content(response_json)
    data = parse_json_object(content)
    return normalize_marking_result(data, expected_max_mark=context.max_mark)


def extract_message_content(response_json: dict[str, Any]) -> str:
    if not isinstance(response_json, dict):
        raise ValueError("Ollama returned a non-object JSON response.")
    message = response_json.get("message")
    if not isinstance(message, dict):
        raise ValueError(f"Ollama response did not include a message object: {response_json}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Ollama response did not include message.content text: {response_json}")
    return content.strip()


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = _parse_first_json_object(content)
    if not isinstance(data, dict):
        raise ValueError(f"Model response was not a JSON object: {content}")
    return data


def normalize_marking_result(data: dict[str, Any], expected_max_mark: float) -> dict[str, Any]:
    total_mark = data.get("total_mark")
    if isinstance(total_mark, bool) or not isinstance(total_mark, (int, float)):
        raise ValueError(f"Model did not return a numeric total_mark: {data}")

    feedback = data.get("overall_feedback") or data.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError(f"Model did not return feedback text: {data}")

    model_max_mark = data.get("max_mark")
    if isinstance(model_max_mark, bool):
        raise ValueError(f"Model returned an invalid max_mark: {data}")

    normalized_total = float(total_mark)
    normalized_max = expected_max_mark

    if isinstance(model_max_mark, (int, float)):
        normalized_model_max = float(model_max_mark)
        if normalized_model_max <= 0:
            raise ValueError(f"Model returned a non-positive max_mark: {data}")
        if abs(normalized_model_max - expected_max_mark) < 1e-9:
            normalized_max = normalized_model_max
        elif abs(normalized_model_max - 100.0) < 1e-9 and 0 <= normalized_total <= 100:
            normalized_total = round((normalized_total * expected_max_mark) / 100.0, 2)
        else:
            raise ValueError(
                "Model returned a max_mark that conflicts with the uploaded marking documents: "
                f"expected {_format_number(expected_max_mark)}, got {_format_number(normalized_model_max)}."
            )
    elif normalized_total > expected_max_mark and normalized_total <= 100 and expected_max_mark != 100:
        normalized_total = round((normalized_total * expected_max_mark) / 100.0, 2)

    if normalized_total < 0 or normalized_total > expected_max_mark:
        raise ValueError(
            "Model returned a total_mark outside the expected range: "
            f"{_format_number(normalized_total)} / {_format_number(expected_max_mark)}."
        )

    return {
        "total_mark": _clean_number(normalized_total),
        "max_mark": _clean_number(normalized_max),
        "overall_feedback": feedback.strip(),
    }


def list_submission_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in {".pdf", ".docx", ".txt"})


def _parse_first_json_object(content: str) -> dict[str, Any]:
    start = content.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(content)):
            char = content[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start : index + 1]
                    return json.loads(candidate)
        start = content.find("{", start + 1)
    raise ValueError(f"No JSON object found in model response: {content}")


def _clean_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 2)


def _format_number(value: float) -> str:
    cleaned = _clean_number(value)
    return str(cleaned)
