import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marking_pipeline.core import (  # noqa: E402
    DEFAULT_OLLAMA_URL,
    MarkingContext,
    ROOT_DIR,
    _normalize_label_key,
    build_assessment_map,
    parse_json_object,
    read_path_text,
)


DEFAULT_CONTEXT_PDF = (
    ROOT_DIR
    / ".scriptgrade_cache"
    / "last_ingest"
    / "marking_scheme"
    / "001_Final_Main_2025_V03_-_solutions.pdf"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "assessment_structure_bakeoff"
EXTRACTION_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
    "num_ctx": 8192,
    "num_predict": 1200,
}
TOP_LEVEL_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
    "num_ctx": 8192,
    "num_predict": 450,
}


TOP_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "instruction_text": {"type": "string"},
                    "max_mark": {"type": ["number", "null"]},
                    "weight_text": {"type": "string"},
                },
                "required": ["label", "instruction_text", "max_mark", "weight_text"],
            },
        }
    },
    "required": ["sections"],
}
SUBPART_SCHEMA = {
    "type": "object",
    "properties": {
        "subparts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "instruction_text": {"type": "string"},
                    "max_mark": {"type": ["number", "null"]},
                    "weight_text": {"type": "string"},
                },
                "required": ["label", "instruction_text", "max_mark", "weight_text"],
            },
        }
    },
    "required": ["subparts"],
}


@dataclass
class MethodResult:
    method: str
    elapsed_seconds: float
    top_level_count: int
    total_unit_count: int
    marks_found_count: int
    labels: list[str]
    sections: list[dict[str, Any]]
    error: str = ""
    notes: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare heuristic, model, and hybrid assessment-structure extraction on marking documents."
    )
    parser.add_argument("--context-pdf", default=str(DEFAULT_CONTEXT_PDF))
    parser.add_argument("--qwen-model", default="qwen2:7b")
    parser.add_argument("--gemma-model", default="gemma3:4b")
    parser.add_argument("--mistral-model", default="mistral:7b")
    parser.add_argument("--llama-model", default="llama3.1:8b")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def build_top_level_messages(document_text: str) -> list[dict[str, str]]:
    system = (
        "Extract only the top-level assessment structure from marking documents. "
        "Return exactly one JSON object matching the schema and no extra text."
    )
    user = (
        "Task:\n"
        "Identify explicit top-level sections such as Part, Question, Section, Task, or Item.\n"
        "Do not invent sections that are not explicit.\n"
        "Do not extract subparts yet.\n"
        "Keep instruction_text faithful to the document.\n"
        "Allocation language may use marks, points, %, weight, weights, weighting, worth, allocated, available, or out of.\n\n"
        "marking_documents:\n"
        f"{document_text}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_subpart_messages(parent_label: str, section_text: str) -> list[dict[str, str]]:
    system = (
        "Extract only the explicit nested subparts inside the provided top-level assessment section. "
        "Return exactly one JSON object matching the schema and no extra text."
    )
    user = (
        f"Top-level section label: {parent_label}\n\n"
        "Task:\n"
        "Identify explicit nested numbered or lettered subparts inside this section only.\n"
        "Do not invent hidden subparts from topic shifts alone.\n"
        "If the section has no explicit subparts, return an empty array.\n"
        "Keep instruction_text faithful to the document.\n"
        "Allocation language may use marks, points, %, weight, weights, weighting, worth, allocated, available, or out of.\n\n"
        "section_text:\n"
        f"{section_text}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_ollama_structured(
    model_name: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    options: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "format": schema,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    response = requests.post(DEFAULT_OLLAMA_URL, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return parse_json_object(content)


def heuristic_sections(context_text: str) -> list[dict[str, Any]]:
    context = MarkingContext(
        rubric_text=context_text,
        brief_text="",
        marking_scheme_text="",
        graded_sample_text="",
        other_context_text="",
        max_mark=100.0,
    )
    assessment_map = build_assessment_map(context)
    top_level: dict[str, dict[str, Any]] = {}
    for unit in assessment_map.units:
        parent_label = unit.parent_label or unit.label
        key = _normalize_label_key(parent_label)
        entry = top_level.setdefault(
            key,
            {
                "label": parent_label,
                "instruction_text": "",
                "max_mark": None,
                "weight_text": "",
                "subparts": [],
            },
        )
        if unit.parent_label:
            entry["subparts"].append(
                {
                    "label": unit.label,
                    "instruction_text": unit.question_text_exact,
                    "max_mark": unit.max_mark,
                    "weight_text": f"{unit.max_mark:g} marks" if unit.max_mark is not None else "",
                }
            )
        else:
            entry["instruction_text"] = unit.question_text_exact
            entry["max_mark"] = unit.max_mark
            entry["weight_text"] = f"{unit.max_mark:g} marks" if unit.max_mark is not None else ""
    return list(top_level.values())


TOP_LEVEL_PATTERN = re.compile(
    r"^(?P<label>(?:part|question|section|task|item)\s+(?:\d+|[ivx]+|[a-z]))\b(?P<tail>.*)$",
    flags=re.IGNORECASE,
)
SUBPART_PATTERN = re.compile(
    r"^(?P<token>(?:\d+|[a-z])(?:[\.\)]|\])|\((?:\d+|[a-z])\))\s*(?P<body>.*)$",
    flags=re.IGNORECASE,
)
MARK_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>marks?|points?|%)\b|\b(?:worth|allocated|allocation|available)\s+(?P<alt>\d+(?:\.\d+)?)\s*(?P<alt_unit>marks?|points?|%)\b",
    flags=re.IGNORECASE,
)


def deterministic_sections(context_text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in context_text.replace("\r\n", "\n").splitlines() if line.strip()]
    sections: list[dict[str, Any]] = []
    current_label = ""
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_label, current_lines
        if not current_label:
            return
        header = current_lines[0] if current_lines else current_label
        body_lines = current_lines[1:] if len(current_lines) > 1 else []
        section = {
            "label": _canonical_top_level_label(current_label),
            "instruction_text": "\n".join(current_lines).strip(),
            "max_mark": extract_max_mark(header),
            "weight_text": extract_weight_text(header),
            "subparts": deterministic_subparts(_canonical_top_level_label(current_label), body_lines),
        }
        if not section["subparts"]:
            section["instruction_text"] = "\n".join(current_lines).strip()
        sections.append(section)
        current_label = ""
        current_lines = []

    for line in lines:
        top_match = TOP_LEVEL_PATTERN.match(line)
        if top_match and is_top_level_header(line, top_match.group("tail")):
            flush_section()
            current_label = top_match.group("label").strip()
            current_lines = [line]
            continue
        if current_label:
            current_lines.append(line)
    flush_section()
    return sections


def deterministic_subparts(parent_label: str, body_lines: list[str]) -> list[dict[str, Any]]:
    if not body_lines:
        return []
    trimmed_lines: list[str] = []
    for line in body_lines:
        if line.lower().startswith("answer:"):
            break
        trimmed_lines.append(line)
    body_lines = trimmed_lines
    if not body_lines:
        return []
    groups: list[list[str]] = []
    current: list[str] = []
    subpart_tokens: list[str] = []
    for line in body_lines:
        sub_match = SUBPART_PATTERN.match(line)
        if sub_match:
            token = normalize_subpart_token(sub_match.group("token"))
            subpart_tokens.append(token)
            if current:
                groups.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        groups.append(current)
    if len(groups) < 2:
        return []
    subparts: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        first_line = group[0]
        sub_match = SUBPART_PATTERN.match(first_line)
        if sub_match is None:
            continue
        token = normalize_subpart_token(sub_match.group("token"))
        label = build_subpart_label(parent_label, token)
        instruction_text = "\n".join(group).strip()
        subparts.append(
            {
                "label": label,
                "instruction_text": instruction_text,
                "max_mark": extract_max_mark(first_line),
                "weight_text": extract_weight_text(first_line),
            }
        )
    return subparts if len(subparts) >= 2 else []


def is_top_level_header(line: str, tail: str) -> bool:
    if extract_max_mark(line) is not None or extract_weight_text(line):
        return True
    normalized_tail = " ".join(tail.split()).strip(" :.-")
    if not normalized_tail:
        return True
    return len(line) <= 32


def normalize_subpart_token(token: str) -> str:
    cleaned = token.strip().strip("()[]").rstrip(".").rstrip(")").strip()
    return cleaned.upper() if cleaned.isalpha() else cleaned


def build_subpart_label(parent_label: str, token: str) -> str:
    if token.isdigit():
        return f"{parent_label} Q{token}"
    return f"{parent_label} {token}"


def _canonical_top_level_label(label: str) -> str:
    parts = label.split()
    if len(parts) < 2:
        return label.strip()
    kind = parts[0].title()
    token = parts[1].upper() if parts[1].isalpha() else parts[1]
    return f"{kind} {token}"


def extract_max_mark(text: str) -> float | None:
    match = MARK_PATTERN.search(text)
    if not match:
        return None
    value = match.group("value") or match.group("alt")
    unit = (match.group("unit") or match.group("alt_unit") or "").lower()
    if not value or unit == "%":
        return None
    return float(value)


def extract_weight_text(text: str) -> str:
    match = MARK_PATTERN.search(text)
    if not match:
        return ""
    value = match.group("value") or match.group("alt") or ""
    unit = match.group("unit") or match.group("alt_unit") or ""
    if not value or not unit:
        return ""
    return f"{value} {unit}".strip()


def normalize_sections(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        normalized.append(
            {
                "label": label,
                "instruction_text": str(item.get("instruction_text", "")).strip(),
                "max_mark": normalize_optional_number(item.get("max_mark")),
                "weight_text": str(item.get("weight_text", "")).strip(),
                "subparts": [],
            }
        )
    return normalized


def normalize_subparts(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        normalized.append(
            {
                "label": label,
                "instruction_text": str(item.get("instruction_text", "")).strip(),
                "max_mark": normalize_optional_number(item.get("max_mark")),
                "weight_text": str(item.get("weight_text", "")).strip(),
            }
        )
    return normalized


def normalize_optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def section_slice(document_text: str, labels: list[str], current_label: str) -> str:
    lowered = document_text.lower()
    starts: list[tuple[int, str]] = []
    for label in labels:
        pos = lowered.find(label.lower())
        if pos != -1:
            starts.append((pos, label))
    if not starts:
        return document_text
    starts.sort()
    current_pos = None
    end_pos = len(document_text)
    for index, (pos, label) in enumerate(starts):
        if label.lower() == current_label.lower():
            current_pos = pos
            if index + 1 < len(starts):
                end_pos = starts[index + 1][0]
            break
    if current_pos is None:
        return document_text
    return document_text[current_pos:end_pos].strip()


def run_staged_model_method(document_text: str, model_name: str) -> tuple[list[dict[str, Any]], float, str, list[str]]:
    started = time.perf_counter()
    notes: list[str] = []
    sections: list[dict[str, Any]] = []
    error = ""

    try:
        top_level_data = call_ollama_structured(
            model_name=model_name,
            messages=build_top_level_messages(document_text),
            schema=TOP_LEVEL_SCHEMA,
            options=TOP_LEVEL_OPTIONS,
            timeout_seconds=120,
        )
        sections = normalize_sections(top_level_data.get("sections"))
        if sections:
            notes.append("top_level_schema_pass")
        else:
            notes.append("top_level_empty")
    except Exception as exc:
        error = f"top_level_failed:{exc}"
        notes.append(error)

    if not sections:
        # Rescue attempt: ask only for labels and marks by keeping the same schema but a shorter slice.
        rescue_text = document_text[:6000]
        try:
            rescue_data = call_ollama_structured(
                model_name=model_name,
                messages=build_top_level_messages(rescue_text),
                schema=TOP_LEVEL_SCHEMA,
                options=TOP_LEVEL_OPTIONS,
                timeout_seconds=90,
            )
            sections = normalize_sections(rescue_data.get("sections"))
            if sections:
                notes.append("top_level_rescue_pass")
                error = ""
            else:
                notes.append("top_level_rescue_empty")
        except Exception as exc:
            notes.append(f"top_level_rescue_failed:{exc}")

    if not sections:
        return [], time.perf_counter() - started, error or "no_sections_detected", notes

    labels = [section["label"] for section in sections]
    for section in sections:
        section_text = section_slice(document_text, labels, section["label"])
        try:
            subpart_data = call_ollama_structured(
                model_name=model_name,
                messages=build_subpart_messages(section["label"], section_text),
                schema=SUBPART_SCHEMA,
                options=EXTRACTION_OPTIONS,
                timeout_seconds=90,
            )
            section["subparts"] = normalize_subparts(subpart_data.get("subparts"))
            if section["subparts"]:
                notes.append(f"subparts_pass:{section['label']}")
        except Exception as exc:
            notes.append(f"subparts_failed:{section['label']}:{exc}")
    return sections, time.perf_counter() - started, error, notes


def merge_hybrid(heuristic: list[dict[str, Any]], model_candidates: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    best_model_sections = max(model_candidates, key=lambda items: (count_units(items), count_marks(items)), default=[])
    model_by_key = {_normalize_label_key(section["label"]): section for section in best_model_sections}
    merged: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for section in heuristic:
        key = _normalize_label_key(section["label"])
        candidate = model_by_key.get(key)
        used_keys.add(key)
        if candidate is None:
            merged.append(section)
            continue
        merged.append(
            {
                "label": section["label"],
                "instruction_text": choose_longer_text(section.get("instruction_text", ""), candidate.get("instruction_text", "")),
                "max_mark": section.get("max_mark") if section.get("max_mark") is not None else candidate.get("max_mark"),
                "weight_text": section.get("weight_text") or candidate.get("weight_text", ""),
                "subparts": merge_hybrid_subparts(section.get("subparts", []), candidate.get("subparts", [])),
            }
        )
    for section in best_model_sections:
        key = _normalize_label_key(section["label"])
        if key not in used_keys:
            merged.append(section)
    return merged


def merge_hybrid_subparts(heuristic_subparts: list[dict[str, Any]], model_subparts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_by_key = {_normalize_label_key(item["label"]): item for item in model_subparts}
    merged: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for item in heuristic_subparts:
        key = _normalize_label_key(item["label"])
        candidate = model_by_key.get(key)
        used_keys.add(key)
        if candidate is None:
            merged.append(item)
            continue
        merged.append(
            {
                "label": item["label"],
                "instruction_text": choose_longer_text(item.get("instruction_text", ""), candidate.get("instruction_text", "")),
                "max_mark": item.get("max_mark") if item.get("max_mark") is not None else candidate.get("max_mark"),
                "weight_text": item.get("weight_text") or candidate.get("weight_text", ""),
            }
        )
    for item in model_subparts:
        key = _normalize_label_key(item["label"])
        if key not in used_keys:
            merged.append(item)
    return merged


def choose_longer_text(first: str, second: str) -> str:
    return second if len(second.strip()) > len(first.strip()) else first


def flatten_labels(sections: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for section in sections:
        labels.append(str(section.get("label", "")).strip())
        for subpart in section.get("subparts", []):
            labels.append(str(subpart.get("label", "")).strip())
    return [label for label in labels if label]


def count_units(sections: list[dict[str, Any]]) -> int:
    return len(flatten_labels(sections))


def count_marks(sections: list[dict[str, Any]]) -> int:
    count = 0
    for section in sections:
        if section.get("max_mark") is not None or section.get("weight_text"):
            count += 1
        for subpart in section.get("subparts", []):
            if subpart.get("max_mark") is not None or subpart.get("weight_text"):
                count += 1
    return count


def evaluate_method(method: str, sections: list[dict[str, Any]], elapsed_seconds: float, error: str = "", notes: list[str] | None = None) -> MethodResult:
    labels = flatten_labels(sections)
    return MethodResult(
        method=method,
        elapsed_seconds=round(elapsed_seconds, 2),
        top_level_count=len(sections),
        total_unit_count=len(labels),
        marks_found_count=count_marks(sections),
        labels=labels,
        sections=sections,
        error=error,
        notes=notes or [],
    )


def build_summary_rows(results: list[MethodResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "elapsed_seconds": result.elapsed_seconds,
                "top_level_count": result.top_level_count,
                "total_unit_count": result.total_unit_count,
                "marks_found_count": result.marks_found_count,
                "labels": " | ".join(result.labels),
                "error": result.error,
                "notes": " | ".join(result.notes or []),
            }
        )
    return rows


def save_workbook(results: list[MethodResult], output_path: Path) -> None:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "summary"
    summary.append(
        ["method", "elapsed_seconds", "top_level_count", "total_unit_count", "marks_found_count", "labels", "error", "notes"]
    )
    for row in build_summary_rows(results):
        summary.append(
            [
                row["method"],
                row["elapsed_seconds"],
                row["top_level_count"],
                row["total_unit_count"],
                row["marks_found_count"],
                row["labels"],
                row["error"],
                row["notes"],
            ]
        )
    for result in results:
        sheet = workbook.create_sheet(result.method[:31])
        sheet.append(
            [
                "top_label",
                "top_max_mark",
                "top_weight_text",
                "top_instruction_text",
                "sub_label",
                "sub_max_mark",
                "sub_weight_text",
                "sub_instruction_text",
            ]
        )
        for section in result.sections:
            subparts = section.get("subparts", [])
            if not subparts:
                sheet.append(
                    [
                        section.get("label", ""),
                        section.get("max_mark"),
                        section.get("weight_text", ""),
                        section.get("instruction_text", ""),
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue
            for index, subpart in enumerate(subparts):
                sheet.append(
                    [
                        section.get("label", "") if index == 0 else "",
                        section.get("max_mark") if index == 0 else "",
                        section.get("weight_text", "") if index == 0 else "",
                        section.get("instruction_text", "") if index == 0 else "",
                        subpart.get("label", ""),
                        subpart.get("max_mark"),
                        subpart.get("weight_text", ""),
                        subpart.get("instruction_text", ""),
                    ]
                )
    workbook.save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    context_text = read_path_text(Path(args.context_pdf)).strip()
    heuristic = heuristic_sections(context_text)
    heuristic_result = evaluate_method("heuristics", heuristic, elapsed_seconds=0.0, notes=["legacy_baseline"])
    deterministic = deterministic_sections(context_text)
    deterministic_result = evaluate_method("deterministic", deterministic, elapsed_seconds=0.0, notes=["generic_parser"])

    model_specs = [
        ("qwen", args.qwen_model),
        ("gemma", args.gemma_model),
        ("mistral", args.mistral_model),
        ("llama", args.llama_model),
    ]
    model_results: list[MethodResult] = []
    model_sections_list: list[list[dict[str, Any]]] = []
    for method_name, model_name in model_specs:
        sections, elapsed, error, notes = run_staged_model_method(context_text, model_name)
        model_sections_list.append(sections)
        model_results.append(evaluate_method(method_name, sections, elapsed, error, notes))

    hybrid_sections = merge_hybrid(deterministic, model_sections_list)
    hybrid_notes = [
        "merge_best_model_with_deterministic",
        f"best_model_units={max((count_units(items) for items in model_sections_list), default=0)}",
    ]
    hybrid_result = evaluate_method(
        "hybrid",
        hybrid_sections,
        elapsed_seconds=max((result.elapsed_seconds for result in model_results), default=0.0),
        notes=hybrid_notes,
    )

    results = [heuristic_result, deterministic_result, *model_results, hybrid_result]
    json_path = output_dir / "assessment_structure_bakeoff.json"
    xlsx_path = output_dir / "assessment_structure_bakeoff.xlsx"
    json_path.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=True, indent=2), encoding="utf-8")
    save_workbook(results, xlsx_path)
    print(json_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
