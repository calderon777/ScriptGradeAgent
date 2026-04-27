"""
Split a full submission text into per-question-part AnswerPart records.

Strategy (configurable via QuestionSplittingConfig):
  - "heuristic": regex-based splitting only
  - "llm":       LLM-based splitting only
  - "hybrid":    heuristic first; fall back to LLM if confidence < threshold
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
    from experiments.cross_sectional_part_marker.src.schemas import QuestionSplittingConfig

from experiments.cross_sectional_part_marker.src.schemas import AnswerPart

logger = logging.getLogger(__name__)

# Regex patterns that identify common question-part headings
_DEFAULT_PATTERNS: list[str] = [
    r"(?i)^part\s+\d+[:\s]?",          # Part 1, Part 1:, PART 2
    r"(?i)^question\s+\d+[a-z]?",      # Question 1, Question 1a
    r"(?i)^Q\d+[a-zA-Z]?[.:\s]",       # Q1. Q1a:
    r"(?i)^\d+[a-zA-Z]\)",             # 1a)
    r"(?i)^\(\s*[a-zA-Z]\s*\)",        # (a)
]

_PART_NUM_RE = re.compile(r"(?i)^part\s+(\d+)")
_QUESTION_NUM_RE = re.compile(r"(?i)(?:question|q)\s*(\d+)")
_PART_LETTER_RE = re.compile(r"(?i)(?:part\s*|[(\s])([a-zA-Z])(?:[).\s]|$)")


def _parse_heading(heading: str) -> tuple[str, str]:
    """
    Attempt to extract (question_id, part_id) from a heading string.
    Returns ("Q?", "") if parsing fails.

    Handles:
      "Part 1:"   -> ("Part_1", "")
      "Question 2a" -> ("Q2", "a")
      "Q3b."      -> ("Q3", "b")
    """
    # Numbered part headings take priority: "Part 1", "PART 2:", etc.
    part_num = _PART_NUM_RE.match(heading.strip())
    if part_num:
        return f"Part_{part_num.group(1)}", ""

    q_match = _QUESTION_NUM_RE.search(heading)
    p_match = _PART_LETTER_RE.search(heading)
    question_id = f"Q{q_match.group(1)}" if q_match else "Q?"
    part_id = p_match.group(1).lower() if p_match else ""
    return question_id, part_id


def split_by_heuristics(
    text: str,
    patterns: list[str] | None = None,
) -> list[tuple[str, str, str, float]]:
    """
    Split *text* on heading-like lines matched by *patterns*.

    Returns
    -------
    list of (heading, question_id, part_id, confidence) tuples.
    heading is the matched line; confidence reflects how cleanly it parsed.
    When no headings are found the whole text is returned as a single chunk.
    """
    compiled = [re.compile(p) for p in (patterns or _DEFAULT_PATTERNS)]
    lines = text.splitlines(keepends=True)

    splits: list[tuple[int, str]] = []  # (line_index, heading_text)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(pat.match(stripped) for pat in compiled):
            splits.append((i, stripped))

    if not splits:
        logger.debug("No question headings found via heuristics")
        return [("", "Q1", "", 0.4)]

    results: list[tuple[str, str, str, float]] = []
    for idx, (line_idx, heading) in enumerate(splits):
        start = line_idx + 1
        end = splits[idx + 1][0] if idx + 1 < len(splits) else len(lines)
        chunk = "".join(lines[start:end]).strip()
        question_id, part_id = _parse_heading(heading)
        confidence = 0.85 if (question_id != "Q?" or part_id) else 0.55
        results.append((chunk, question_id, part_id, confidence))

    return results


def split_by_llm(
    text: str,
    client: "OllamaClient",
    model: str,
    temperature: float = 0.1,
) -> list[tuple[str, str, str, float]]:
    """
    Ask a local LLM to identify question-part boundaries in *text*.

    Returns the same (chunk, question_id, part_id, confidence) list.
    """
    prompt = (
        "You are a marking assistant. The following text is a student exam submission.\n"
        "Identify each question/part boundary and return JSON in this exact format:\n"
        '{"parts": [{"question_id": "Q1", "part_id": "a", "text": "...student answer text..."}]}\n\n'
        "Do not include any explanation — only the JSON.\n\n"
        f"SUBMISSION:\n{text[:6000]}"
    )
    raw = client.generate(model=model, prompt=prompt, temperature=temperature)
    try:
        data = json.loads(raw)
        parts = data.get("parts", [])
        return [
            (p.get("text", ""), p.get("question_id", "Q?"), p.get("part_id", ""), 0.75)
            for p in parts
        ]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("LLM split returned unparseable JSON: %s", exc)
        return [("", "Q1", "", 0.3)]


def build_answer_parts(
    submission_id: str,
    anonymised_student_id: str,
    source_file: str,
    text: str,
    config: "QuestionSplittingConfig",
    client: "OllamaClient | None" = None,
    model: str = "qwen2.5:7b",
    temperature: float = 0.1,
) -> list[AnswerPart]:
    """
    Split *text* and return a list of AnswerPart records.

    Parameters
    ----------
    submission_id:          Unique ID for this submission
    anonymised_student_id:  Anonymised student ID
    source_file:            Path to source file (string)
    text:                   Full submission text
    config:                 QuestionSplittingConfig
    client:                 OllamaClient (required for llm/hybrid modes)
    model:                  Model name for LLM splitting
    temperature:            Sampling temperature
    """
    method = config.method
    threshold = config.llm_fallback_threshold
    patterns = config.patterns or _DEFAULT_PATTERNS

    chunks: list[tuple[str, str, str, float]] = []

    if method == "heuristic":
        chunks = split_by_heuristics(text, patterns)
    elif method == "llm":
        if client is None:
            logger.error("LLM splitting requested but no OllamaClient provided — falling back to heuristic")
            chunks = split_by_heuristics(text, patterns)
        else:
            chunks = split_by_llm(text, client, model, temperature)
    elif method == "hybrid":
        chunks = split_by_heuristics(text, patterns)
        min_conf = min(c[3] for c in chunks) if chunks else 0.0
        if min_conf < threshold and client is not None:
            logger.info(
                "Heuristic confidence %.2f below threshold %.2f — using LLM fallback",
                min_conf,
                threshold,
            )
            llm_chunks = split_by_llm(text, client, model, temperature)
            if llm_chunks and llm_chunks[0][3] > min_conf:
                chunks = llm_chunks

    if not chunks:
        logger.warning("No chunks produced for submission %s; treating as single part", submission_id)
        chunks = [(text, "Q1", "", 0.3)]

    parts: list[AnswerPart] = []
    seen: dict[tuple[str, str], int] = {}  # deduplicate Q+part combos
    for chunk_text, question_id, part_id, confidence in chunks:
        # Deduplicate: if we've already seen this (Q, part) pair, suffix with counter
        key = (question_id, part_id)
        if key in seen:
            seen[key] += 1
            part_id = f"{part_id}_{seen[key]}"
        else:
            seen[key] = 0

        notes: list[str] = []
        if confidence < 0.6:
            notes.append("Low-confidence split — may need human review")
        if question_id == "Q?":
            notes.append("Could not parse question number from heading")

        cleaned = _clean_text(chunk_text)
        parts.append(
            AnswerPart(
                submission_id=submission_id,
                anonymised_student_id=anonymised_student_id,
                source_file=source_file,
                question_id=question_id,
                part_id=part_id,
                raw_text=chunk_text,
                cleaned_text=cleaned,
                word_count=len(cleaned.split()),
                extraction_confidence=round(confidence, 3),
                extraction_notes="; ".join(notes),
            )
        )

    return parts


def _clean_text(text: str) -> str:
    """Light cleaning: collapse excessive whitespace, strip leading/trailing space."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
