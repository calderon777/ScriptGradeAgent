import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf4llm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marking_pipeline.core import (  # noqa: E402
    MarkingContext,
    ROOT_DIR,
    call_ollama,
    clear_prepared_assessment_map_cache,
    prepare_assessment_map,
    read_path_text,
)


ASSESSMENT_CONTEXT_PDF = (
    ROOT_DIR
    / ".scriptgrade_cache"
    / "last_ingest"
    / "marking_scheme"
    / "001_Final_Main_2025_V03_-_solutions.pdf"
)
OUTPUT_DIR = ROOT_DIR / "output" / "end_to_end_extractor_bakeoff"
MODEL_NAME = "qwen2:7b"
USE_VERIFIER = False


@dataclass
class EndToEndResult:
    backend: str
    path: str
    extraction_seconds: float
    extracted_word_count: int
    status: str
    total_mark: float | None = None
    max_mark: float | None = None
    detected_part_count: int | None = None
    detected_part_labels: list[str] | None = None
    latency_total: float | None = None
    latency_structure_detect: float | None = None
    latency_part_refine: float | None = None
    latency_part_analysis: float | None = None
    latency_moderation: float | None = None
    latency_finalize: float | None = None
    validation_notes: list[str] | None = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare end-to-end grading across extraction backends.")
    parser.add_argument("--source-root", help="Root folder to sample PDFs from.")
    parser.add_argument("--sample-list", help="Optional text file containing one PDF path per line.")
    parser.add_argument("--sample-count", type=int, default=2, help="Number of PDFs to sample.")
    parser.add_argument("--seed", type=int, default=3040, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def load_sample_paths(args: argparse.Namespace) -> list[Path]:
    if args.sample_list:
        sample_list = Path(args.sample_list)
        return [Path(line.strip()) for line in sample_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not args.source_root:
        raise ValueError("Provide either --sample-list or --source-root.")
    source_root = Path(args.source_root)
    pdf_paths = sorted(source_root.rglob("*.pdf"))
    if len(pdf_paths) < args.sample_count:
        raise ValueError(f"Requested {args.sample_count} PDFs but found only {len(pdf_paths)} under {source_root}.")
    random.seed(args.seed)
    return random.sample(pdf_paths, args.sample_count)


def load_context() -> MarkingContext:
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


def extract_pymupdf4llm_clean(path: Path) -> str:
    return cleanup_pymupdf4llm_text(pymupdf4llm.to_markdown(str(path)))


def run_backend(path: Path, backend: str, extractor, context: MarkingContext, prepared_assessment_map) -> EndToEndResult:
    started = time.perf_counter()
    script_text = extractor(path)
    extraction_seconds = round(time.perf_counter() - started, 2)
    try:
        result = call_ollama(
            script_text=script_text,
            context=context,
            filename=path.name,
            model_name=MODEL_NAME,
            rubric_verifier_model_name=MODEL_NAME if USE_VERIFIER else None,
            prepared_assessment_map=prepared_assessment_map,
        )
        return EndToEndResult(
            backend=backend,
            path=str(path),
            extraction_seconds=extraction_seconds,
            extracted_word_count=len(script_text.split()),
            status="ok",
            total_mark=result.get("total_mark"),
            max_mark=result.get("max_mark"),
            detected_part_count=result.get("detected_part_count"),
            detected_part_labels=list(result.get("detected_part_labels", [])),
            latency_total=result.get("latency_seconds_total"),
            latency_structure_detect=result.get("latency_seconds_structure_detect"),
            latency_part_refine=result.get("latency_seconds_part_refine"),
            latency_part_analysis=result.get("latency_seconds_part_analysis"),
            latency_moderation=result.get("latency_seconds_moderation"),
            latency_finalize=result.get("latency_seconds_finalize"),
            validation_notes=list(result.get("validation_notes", [])),
        )
    except Exception as exc:
        return EndToEndResult(
            backend=backend,
            path=str(path),
            extraction_seconds=extraction_seconds,
            extracted_word_count=len(script_text.split()),
            status="error",
            error=str(exc),
        )


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_paths = load_sample_paths(args)
    (OUTPUT_DIR / "sample_paths.txt").write_text("\n".join(str(path) for path in sample_paths), encoding="utf-8")
    context = load_context()
    clear_prepared_assessment_map_cache()
    prepared_assessment_map = prepare_assessment_map(context=context, verifier_model_name=None)
    backends = [
        ("current", extract_current),
        ("pymupdf4llm_clean", extract_pymupdf4llm_clean),
    ]
    results: list[EndToEndResult] = []
    for path in sample_paths:
        print(f"\nFILE::{path}")
        for backend_name, extractor in backends:
            row = run_backend(path, backend_name, extractor, context, prepared_assessment_map)
            results.append(row)
            print(json.dumps(asdict(row), ensure_ascii=True))

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
