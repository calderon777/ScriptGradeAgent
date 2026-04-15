import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf4llm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marking_pipeline.core import (  # noqa: E402
    DEFAULT_OLLAMA_URL,
    ROOT_DIR,
    MarkingContext,
    _call_ollama_json,
    build_structure_messages_with_guidance,
    extract_expected_parts_from_context,
    extract_structure_guidance,
    normalize_detected_parts,
    prepare_marking_context,
    read_path_text,
)


SAMPLE_LIST = ROOT_DIR / "output" / "extractor_bakeoff_ec3040" / "sample_paths.txt"
ASSESSMENT_CONTEXT_PDF = (
    ROOT_DIR
    / ".scriptgrade_cache"
    / "last_ingest"
    / "marking_scheme"
    / "001_Final_Main_2025_V03_-_solutions.pdf"
)
OUTPUT_DIR = ROOT_DIR / "output" / "section_detection_bakeoff"
MODEL_NAME = "qwen2:7b"


@dataclass
class DetectionResult:
    backend: str
    path: str
    elapsed_seconds: float
    detected_count: int
    matched_expected_labels: int
    expected_count: int
    exact_match: bool
    detected_labels: list[str]
    expected_labels: list[str]
    anchor_hits_in_text: int
    extracted_word_count: int
    error: str = ""


def load_sample_paths() -> list[Path]:
    return [Path(line.strip()) for line in SAMPLE_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_context():
    rubric_text = read_path_text(ASSESSMENT_CONTEXT_PDF)
    return MarkingContext(
        rubric_text=rubric_text.strip(),
        brief_text="",
        marking_scheme_text="",
        graded_sample_text="",
        other_context_text="",
        max_mark=100.0,
    )


def extract_current(path: Path) -> str:
    return read_path_text(path)


def extract_pymupdf4llm_clean(path: Path) -> str:
    raw = pymupdf4llm.to_markdown(str(path))
    return cleanup_pymupdf4llm_text(raw)


def cleanup_pymupdf4llm_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.replace("\f", "\n").splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if "picture [" in stripped and "intentionally omitted" in stripped:
            continue
        if stripped.isdigit():
            continue
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = stripped.replace("**", "").replace("__", "")
        stripped = stripped.replace("_", "")
        stripped = stripped.replace("`", "")
        stripped = stripped.replace("<br>", " ")
        if "|" in stripped and stripped.count("|") >= 2:
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
            stripped = " ".join(cells)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.replace("Part1", "Part 1").replace("Question1", "Question 1")
    return cleaned.strip()


def run_detection(path: Path, backend: str, extracted_text: str, expected_labels: list[str], structure_guidance: str) -> DetectionResult:
    started = time.perf_counter()
    try:
        data = _call_ollama_json(
            model_name=MODEL_NAME,
            messages=build_structure_messages_with_guidance(extracted_text, path.name, structure_guidance),
            ollama_url=DEFAULT_OLLAMA_URL,
        )
        parts = normalize_detected_parts(data)
        elapsed = round(time.perf_counter() - started, 2)
        detected_labels = [part.label for part in parts]
        matched = sum(1 for label in detected_labels if label in expected_labels)
        anchor_hits = sum(1 for part in parts if part.anchor_text and part.anchor_text in extracted_text)
        return DetectionResult(
            backend=backend,
            path=str(path),
            elapsed_seconds=elapsed,
            detected_count=len(detected_labels),
            matched_expected_labels=matched,
            expected_count=len(expected_labels),
            exact_match=detected_labels == expected_labels,
            detected_labels=detected_labels,
            expected_labels=expected_labels,
            anchor_hits_in_text=anchor_hits,
            extracted_word_count=len(extracted_text.split()),
        )
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 2)
        return DetectionResult(
            backend=backend,
            path=str(path),
            elapsed_seconds=elapsed,
            detected_count=0,
            matched_expected_labels=0,
            expected_count=len(expected_labels),
            exact_match=False,
            detected_labels=[],
            expected_labels=expected_labels,
            anchor_hits_in_text=0,
            extracted_word_count=len(extracted_text.split()),
            error=str(exc),
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_paths = load_sample_paths()
    context = load_context()
    structure_guidance = extract_structure_guidance(context)
    expected_labels = [part.label for part in extract_expected_parts_from_context(context)]

    results: list[DetectionResult] = []
    backends = [
        ("current", extract_current),
        ("pymupdf4llm_clean", extract_pymupdf4llm_clean),
    ]

    for path in sample_paths:
        print(f"\nFILE::{path}")
        for backend_name, backend_func in backends:
            extracted_text = backend_func(path)
            result = run_detection(path, backend_name, extracted_text, expected_labels, structure_guidance)
            results.append(result)
            print(json.dumps(asdict(result), ensure_ascii=True))

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
