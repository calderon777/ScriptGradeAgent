"""
Ingest submission files (PDF or .txt) and extract per-question-part AnswerPart records.

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.ingest \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

Outputs:
    <output_folder>/answer_parts.jsonl  — one AnswerPart JSON per line
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
from experiments.cross_sectional_part_marker.src.part_splitter import build_answer_parts
from experiments.cross_sectional_part_marker.src.schemas import AnswerPart, PipelineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract all text from a PDF using pypdf (preferred) or PyMuPDF as fallback.
    Returns the concatenated text of all pages.
    """
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        logger.debug("pypdf extracted %d chars from %s", len(text), path.name)
        return text
    except ImportError:
        logger.warning("pypdf not available, trying PyMuPDF")
    except Exception as exc:
        logger.warning("pypdf failed on %s: %s — trying PyMuPDF", path.name, exc)

    try:
        import fitz  # PyMuPDF  # type: ignore

        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages)
        logger.debug("PyMuPDF extracted %d chars from %s", len(text), path.name)
        return text
    except ImportError:
        raise RuntimeError("Neither pypdf nor PyMuPDF is installed. Cannot read PDFs.")
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from {path}: {exc}") from exc


def extract_text_from_txt(path: Path) -> str:
    """Read a plain-text submission file."""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text_from_docx(path: Path) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        import docx  # type: ignore
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n\n".join(paragraphs)
        logger.debug("python-docx extracted %d chars from %s", len(text), path.name)
        return text
    except ImportError:
        raise RuntimeError("python-docx is not installed. Cannot read .docx files.")
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from {path}: {exc}") from exc


def extract_text(path: Path) -> str:
    """Dispatch text extraction based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in (".txt", ".md"):
        return extract_text_from_txt(path)
    elif suffix == ".docx":
        return extract_text_from_docx(path)
    else:
        logger.warning("Unsupported file type %s; attempting plain-text read", suffix)
        return extract_text_from_txt(path)


def _derive_submission_id(path: Path) -> str:
    """Derive a submission ID from filename stem, cleaned to safe chars."""
    return re.sub(r"[^\w\-]", "_", path.stem)


def _derive_student_id(path: Path) -> str:
    """
    Derive an anonymised student ID.
    Checks parent folder name first (Moodle pattern: 'Name_1234567_assignsubmission_file'),
    then falls back to numeric run in the filename stem, then the stem itself.
    """
    # Try parent folder: look for a 7-digit Moodle participant ID (e.g. Name_1494034_assignsubmission_file)
    folder_nums = re.findall(r"(?<!\d)(\d{7})(?!\d)", path.parent.name)
    if folder_nums:
        return f"ANON_{folder_nums[0]}"
    # Fallback: any number in filename
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return f"ANON_{nums[0]}"
    return f"ANON_{path.stem}"


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------


def run_ingest(config: PipelineConfig, force: bool = False) -> Path:
    """
    Ingest all submission files in config.submissions_folder.

    Parameters
    ----------
    config: Loaded PipelineConfig
    force:  If True, overwrite existing answer_parts.jsonl

    Returns
    -------
    Path to the written answer_parts.jsonl
    """
    submissions_dir = Path(config.submissions_folder)
    output_dir = Path(config.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "answer_parts.jsonl"

    if output_path.exists() and not force:
        logger.info("answer_parts.jsonl already exists — skipping (use --force to overwrite)")
        return output_path

    if not submissions_dir.exists():
        raise FileNotFoundError(f"Submissions folder not found: {submissions_dir}")

    submission_files = sorted(
        list(submissions_dir.glob("**/*.pdf"))
        + list(submissions_dir.glob("**/*.txt"))
        + list(submissions_dir.glob("**/*.docx"))
    )
    if not submission_files:
        logger.warning("No .pdf or .txt files found in %s", submissions_dir)
        return output_path

    logger.info("Found %d submission files in %s", len(submission_files), submissions_dir)

    # Build OllamaClient if hybrid/llm splitting is needed
    client: OllamaClient | None = None
    if config.question_splitting.method in ("llm", "hybrid"):
        client = OllamaClient(
            base_url=config.ollama.base_url,
            timeout=config.ollama.timeout,
            stage="ingest",
        )

    all_parts: list[AnswerPart] = []
    for sub_file in submission_files:
        submission_id = _derive_submission_id(sub_file)
        student_id = _derive_student_id(sub_file)
        logger.info("Processing submission: %s", sub_file.name)

        try:
            text = extract_text(sub_file)
        except Exception as exc:
            logger.error("Failed to extract text from %s: %s — skipping", sub_file.name, exc)
            continue

        if not text.strip():
            logger.warning("Empty text extracted from %s — skipping", sub_file.name)
            continue

        parts = build_answer_parts(
            submission_id=submission_id,
            anonymised_student_id=student_id,
            source_file=str(sub_file),
            text=text,
            config=config.question_splitting,
            client=client,
            model=config.models.analysis_model,
            temperature=config.temperature,
        )

        for part in parts:
            if part.extraction_confidence < 0.7:
                logger.warning(
                    "Low extraction confidence (%.2f) for %s %s%s: %s",
                    part.extraction_confidence,
                    part.submission_id,
                    part.question_id,
                    part.part_id,
                    part.extraction_notes,
                )
        all_parts.extend(parts)

    # Write output
    with open(output_path, "w", encoding="utf-8") as fh:
        for part in all_parts:
            fh.write(part.model_dump_json() + "\n")

    logger.info("Wrote %d answer parts to %s", len(all_parts), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest submission files into answer_parts.jsonl")
    p.add_argument("--config", required=True, help="Path to YAML pipeline config")
    p.add_argument("--force", action="store_true", help="Overwrite existing output")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    import sys

    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    cfg = PipelineConfig.model_validate(raw_cfg)

    out = run_ingest(cfg, force=args.force)
    print(f"Ingest complete: {out}")
    sys.exit(0)
