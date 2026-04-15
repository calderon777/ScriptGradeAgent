import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz
import pymupdf4llm
from markitdown import MarkItDown

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marking_pipeline.core import ROOT_DIR, read_path_text


PDF_PATHS = [
    ROOT_DIR / "scripts" / "test" / "student1.pdf",
    ROOT_DIR / ".scriptgrade_cache" / "last_ingest" / "marking_scheme" / "001_Final_Main_2025_V03_-_solutions.pdf",
]
OUTPUT_DIR = ROOT_DIR / "output" / "extractor_bakeoff"


@dataclass
class ExtractionResult:
    extractor: str
    path: str
    elapsed_seconds: float
    char_count: int
    word_count: int
    line_count: int
    heading_like_lines: int
    suspicious_char_count: int
    empty: bool
    error: str = ""


def extract_current(path: Path) -> str:
    return read_path_text(path)


def extract_markitdown(path: Path) -> str:
    converter = MarkItDown()
    result = converter.convert(str(path))
    return getattr(result, "text_content", "") or ""


def extract_pymupdf4llm(path: Path) -> str:
    return pymupdf4llm.to_markdown(str(path))


def count_heading_like_lines(text: str) -> int:
    heading_patterns = (
        r"^#{1,6}\s",
        r"^[A-Z][A-Z0-9 ,:()/_-]{6,}$",
        r"^\d+(\.\d+)*\s+[A-Z]",
        r"^(Question|Part|Section)\b",
    )
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(re.search(pattern, stripped) for pattern in heading_patterns):
            count += 1
    return count


def count_suspicious_chars(text: str) -> int:
    return sum(text.count(char) for char in ("�", "\ufffd", "â€", "â€“", "\x00"))


def summarize_text(text: str) -> tuple[int, int, int, int]:
    stripped = text.strip()
    words = stripped.split()
    lines = [line for line in stripped.splitlines() if line.strip()]
    return len(stripped), len(words), len(lines), count_heading_like_lines(stripped)


def run_extractor(name: str, func, path: Path) -> tuple[ExtractionResult, str]:
    started = time.perf_counter()
    try:
        text = func(path)
        elapsed = round(time.perf_counter() - started, 2)
        char_count, word_count, line_count, heading_like_lines = summarize_text(text)
        result = ExtractionResult(
            extractor=name,
            path=str(path),
            elapsed_seconds=elapsed,
            char_count=char_count,
            word_count=word_count,
            line_count=line_count,
            heading_like_lines=heading_like_lines,
            suspicious_char_count=count_suspicious_chars(text),
            empty=not bool(text.strip()),
        )
        return result, text
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 2)
        result = ExtractionResult(
            extractor=name,
            path=str(path),
            elapsed_seconds=elapsed,
            char_count=0,
            word_count=0,
            line_count=0,
            heading_like_lines=0,
            suspicious_char_count=0,
            empty=True,
            error=str(exc),
        )
        return result, ""


def save_output(path: Path, extractor: str, text: str) -> None:
    stem = path.stem
    output_path = OUTPUT_DIR / f"{stem}.{extractor}.md"
    output_path.write_text(text, encoding="utf-8")


def save_summary(results: list[ExtractionResult]) -> None:
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractors = [
        ("current", extract_current),
        ("markitdown", extract_markitdown),
        ("pymupdf4llm", extract_pymupdf4llm),
    ]
    results: list[ExtractionResult] = []

    for path in PDF_PATHS:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        page_count = fitz.open(path).page_count
        print(f"\n== {path} ({page_count} pages) ==")
        for extractor_name, extractor_func in extractors:
            result, text = run_extractor(extractor_name, extractor_func, path)
            results.append(result)
            print(asdict(result))
            if not result.error:
                save_output(path, extractor_name, text)

    save_summary(results)


if __name__ == "__main__":
    main()
