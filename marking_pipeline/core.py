import io
import hashlib
import json
import re
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pdfplumber
import pymupdf
import requests
from docx import Document
try:
    from docx2python import docx2python
except ImportError:  # pragma: no cover - optional dependency
    docx2python = None
try:
    import pymupdf4llm
except ImportError:  # pragma: no cover - optional dependency
    pymupdf4llm = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
SUPPORTED_TEXT_SUFFIXES = {".pdf", ".docx", ".txt"}

DEFAULT_CONTEXT_PATTERNS = {
    "rubric": ("rubric",),
    "brief": ("brief", "assignment"),
    "marking_scheme": ("marking_scheme", "marking scheme", "scheme"),
    "graded_sample": ("graded_sample", "graded sample", "sample", "example"),
    "other": ("support", "context", "guidance", "instruction"),
}


@dataclass(frozen=True)
class MarkingContext:
    rubric_text: str
    brief_text: str
    marking_scheme_text: str
    graded_sample_text: str
    other_context_text: str
    max_mark: float


@dataclass(frozen=True)
class AssessmentBundle:
    name: str
    folder: Path
    submission_files: tuple[Path, ...]
    rubric_files: tuple[Path, ...]
    brief_files: tuple[Path, ...]
    marking_scheme_files: tuple[Path, ...]
    graded_sample_files: tuple[Path, ...]
    other_files: tuple[Path, ...]


@dataclass(frozen=True)
class SubmissionPart:
    label: str
    focus_hint: str = ""
    anchor_text: str = ""
    section_text: str = ""
    max_mark: float | None = None
    marking_guidance: str = ""
    question_text_exact: str = ""
    task_type: str = ""


@dataclass(frozen=True)
class SubmissionDiagnostics:
    extracted_word_count: int
    detected_part_count: int
    detected_part_labels: tuple[str, ...]
    low_text: bool
    possible_extraction_issue: bool


@dataclass(frozen=True)
class AssessmentUnit:
    label: str
    max_mark: float | None = None
    parent_label: str = ""
    grading_mode: str = "deterministic"
    task_type: str = ""
    dependency_group: str = ""
    question_text_exact: str = ""
    marking_guidance: str = ""
    rubric_text: str = ""
    rubric_confidence_0_to_100: float | None = None
    rubric_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssessmentMap:
    units: tuple[AssessmentUnit, ...]
    overall_max_mark: float
    scale_confidence_0_to_100: float


@dataclass(frozen=True)
class PreparedAssessmentMap:
    cache_key: str
    original_map: AssessmentMap
    prepared_map: AssessmentMap
    prepared_units: tuple["PreparedAssessmentUnit", ...] = ()
    verifier_model_name: str | None = None
    verification_applied: bool = False
    verifier_attempt_count: int = 0
    preparation_issues: tuple[str, ...] = ()
    preparation_confidence_0_to_100: float | None = None
    preparation_latency_seconds: float = 0.0


@dataclass(frozen=True)
class PreparedAssessmentUnit:
    label: str
    compact_criteria: tuple[str, ...] = ()
    verifier_confidence_0_to_100: float | None = None
    verifier_issues: tuple[str, ...] = ()
    verifier_attempt_count: int = 0
    verifier_accepted_refinements: tuple[str, ...] = ()
    verifier_rejected_refinements: tuple[str, ...] = ()


_PREPARED_ASSESSMENT_CACHE: dict[str, PreparedAssessmentMap] = {}
MAX_PREPARED_MAP_VERIFIER_ATTEMPTS = 3
MIN_ACCEPTABLE_PREPARATION_CONFIDENCE = 75.0


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
    raw = file_obj.read()
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    xml_text = _extract_text_from_docx_xml(raw)
    if xml_text.strip():
        return xml_text.strip()
    doc = Document(io.BytesIO(raw))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()


def extract_docx_structure_text(raw: bytes) -> str:
    doc = Document(io.BytesIO(raw))
    parts: list[str] = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(f"{_docx_heading_marker(paragraph)}{text}")
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts).strip()


def _docx_heading_marker(paragraph: Any) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text)
    if len(normalized) > 140:
        return ""
    text_runs = [run for run in paragraph.runs if run.text.strip()]
    if not text_runs:
        return ""
    bold_runs = sum(1 for run in text_runs if bool(run.bold))
    size_values = [run.font.size.pt for run in text_runs if run.font.size is not None]
    max_size = max(size_values) if size_values else None
    style_name = (getattr(getattr(paragraph, "style", None), "name", "") or "").lower()
    heading_like_text = bool(re.match(r"^(part|question|section|appendix)\b", normalized, re.IGNORECASE))
    compact_question_like = bool(re.match(r"^(q(?:uestion)?\s*\d+[a-z]?|[ivxlcdm]+[\.\)]|(?:\d+|[A-Z])[\.\)])\b", normalized, re.IGNORECASE))
    title_case_like = bool(re.match(r"^[A-Z][A-Za-z0-9 ,:/&()'\"-]{0,90}$", normalized))
    all_caps_like = normalized.isupper() and len(normalized.split()) <= 12
    heading_signal = (
        heading_like_text
        or "heading" in style_name
        or (compact_question_like and len(normalized.split()) <= 12)
        or (title_case_like and (bold_runs == len(text_runs) or max_size is not None and max_size >= 13))
        or all_caps_like
        or (bold_runs == len(text_runs) and len(normalized.split()) <= 16)
        or (max_size is not None and max_size >= 14 and len(normalized.split()) <= 20)
    )
    return "[HEADING] " if heading_signal else ""


def choose_docx_scoring_text(primary_text: str, fallback_text: str) -> str:
    primary = primary_text.strip()
    fallback = fallback_text.strip()
    if not primary:
        return fallback
    latex_hits = primary.count("<latex>")
    media_hits = primary.count("----media/")
    if latex_hits > 0 and media_hits <= max(2, latex_hits // 8):
        return primary
    if media_hits > 0 and latex_hits == 0:
        return fallback or primary
    return primary or fallback


def extract_docx_scoring_text(raw: bytes) -> str:
    fallback_text = extract_text_from_docx(io.BytesIO(raw))
    if docx2python is None:
        return fallback_text
    with docx2python(io.BytesIO(raw)) as result:
        text = (result.text or "").strip()
    return choose_docx_scoring_text(text, fallback_text)


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
        stripped = stripped.replace("`", "")
        stripped = stripped.replace("<br>", " ")
        if "|" in stripped and stripped.count("|") >= 2:
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
            stripped = " ".join(cells)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        cleaned_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pdf_structure_text(raw: bytes) -> str:
    if pymupdf4llm is None:
        return ""
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        extracted = pymupdf4llm.to_markdown(doc, header=False, footer=False)
    finally:
        doc.close()
    return cleanup_pymupdf4llm_text(extracted)


def build_submission_texts_from_path(path: Path) -> tuple[str, str | None]:
    if path.suffix.lower() == ".docx":
        raw = path.read_bytes()
        script_text = extract_docx_scoring_text(raw)
        structure_text = extract_docx_structure_text(raw) or None
        return script_text, structure_text

    script_text = read_path_text(path)
    structure_text: str | None = None
    if path.suffix.lower() == ".pdf":
        structure_text = extract_pdf_structure_text(path.read_bytes()) or None
    return script_text, structure_text


def build_submission_texts_from_upload(uploaded_file: Any) -> tuple[str, str | None]:
    raw = uploaded_file.getvalue()
    lowered = uploaded_file.name.lower()
    if lowered.endswith(".docx"):
        script_text = extract_docx_scoring_text(raw)
        structure_text = extract_docx_structure_text(raw) or None
        return script_text, structure_text

    script_text = read_file_bytes(uploaded_file.name, raw)
    structure_text: str | None = None
    if lowered.endswith(".pdf"):
        structure_text = extract_pdf_structure_text(raw) or None
    return script_text, structure_text


def read_paths_text(paths: list[Path] | tuple[Path, ...]) -> str:
    if not paths:
        return ""
    return "\n\n".join(text for text in (read_path_text(path) for path in paths) if text).strip()


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


def build_part_messages(
    part: SubmissionPart,
    context: MarkingContext,
    filename: str,
) -> list[dict[str, str]]:
    max_mark_text = _format_number(part.max_mark) if part.max_mark is not None else "100"
    score_key = '- "provisional_score": number\n'
    score_rule = f"1. provisional_score must be between 0 and {max_mark_text}.\n"
    if part.max_mark is None:
        score_key = '- "provisional_score_0_to_100": number\n'
        score_rule = "1. provisional_score_0_to_100 must be between 0 and 100.\n"
    task_text = _build_part_task_text(part)
    criteria = _extract_prompt_criteria(part.marking_guidance)
    if not criteria:
        criteria = _default_prompt_criteria(task_text)
    scoring_rules = _extract_prompt_scoring_rules(part.marking_guidance, part.max_mark)
    criteria_block = "\n".join(f"- {criterion}" for criterion in criteria)
    scoring_rule_lines = [score_rule.rstrip(), "2. Base the score only on this section text and the section-specific criteria."]
    for index, rule in enumerate(scoring_rules, start=3):
        scoring_rule_lines.append(f"{index}. {rule}")
    system = (
        "You are a careful university examiner working in stages. "
        "Assess only the supplied section of the student's submission. "
        "Use only the section task, the section-specific scoring criteria, and the section text."
    )
    user = (
        "You are assessing one section of a student's script.\n\n"
        f"STUDENT FILE NAME: {filename}\n"
        f"SECTION LABEL: {part.label}\n"
        f"SECTION MAX MARK: {max_mark_text}\n"
        f"SECTION TASK: {task_text}\n\n"
        "SCORING CRITERIA:\n"
        f"{criteria_block}\n\n"
        "SECTION TEXT:\n"
        f"\"\"\"{part.section_text or part.focus_hint or part.label}\"\"\"\n\n"
        "Return only one JSON object with exactly these keys:\n"
        '- "section_label": string\n'
        f"{score_key}"
        '- "grade_band": string\n'
        '- "within_band_position": string\n'
        '- "band_confidence_0_to_100": number\n'
        '- "strengths": array of strings\n'
        '- "weaknesses": array of strings\n'
        '- "evidence": array of strings\n'
        '- "coverage_comment": string\n\n'
        "Rules:\n"
        + "\n".join(scoring_rule_lines)
        + "\n"
        + "6. strengths must contain at least 2 concrete items.\n"
        + "7. weaknesses must contain at least 2 concrete items.\n"
        + "8. evidence must contain concrete references to the section content.\n"
        + "9. coverage_comment must state the classification judgement for this section and why the mark sits where it does within the bracket.\n"
        + "10. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _build_part_task_text(part: SubmissionPart) -> str:
    task_from_type = _task_goal_from_task_type(part.task_type, part.question_text_exact)
    if task_from_type:
        return task_from_type
    focus = part.focus_hint.strip()
    guidance_focus = _extract_task_focus_from_guidance(part.marking_guidance)
    if focus:
        return focus
    if guidance_focus:
        return guidance_focus
    return "Assess the content that best matches this section label."


def _extract_task_focus_from_guidance(marking_guidance: str) -> str:
    text = marking_guidance.strip()
    if not text:
        return ""
    match = re.search(r"Task focus:\s*(.+)", text)
    if match is not None:
        return match.group(1).strip().rstrip(".")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith(("unit:", "available marks:", "compact scoring criteria:", "detailed rubric:")):
            continue
        if lowered.startswith("use this rubric") or lowered.startswith("use uk-style quality bands"):
            continue
        if stripped.startswith("- "):
            continue
        return stripped.rstrip(".")
    return ""


def _extract_prompt_criteria(marking_guidance: str) -> list[str]:
    text = marking_guidance.strip()
    if not text:
        return []
    criteria: list[str] = []
    in_compact_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_compact_block and criteria:
                break
            continue
        lowered = stripped.lower()
        if lowered == "compact scoring criteria:":
            in_compact_block = True
            continue
        if lowered.startswith("detailed rubric:"):
            if criteria:
                break
            in_compact_block = False
            continue
        if not stripped.startswith("- "):
            continue
        item = stripped[2:].strip()
        if not item:
            continue
        if in_compact_block:
            criteria.append(item)
        elif lowered.startswith(("- place work", "- use the top band", "- award explicit partial credit", "- rank attempts")):
            break
        elif item.lower().startswith(("evaluate whether", "assess whether", "check whether", "identify whether")):
            criteria.append(item)
        if len(criteria) >= 5:
            break
    if criteria:
        return criteria
    if "\n" not in text and len(text.split()) <= 40:
        return [text.rstrip(".") + "."]
    return criteria


def _default_prompt_criteria(task_text: str) -> list[str]:
    return [
        f"Evaluate whether the response directly addresses {task_text.lower()}.",
        "Evaluate whether the reasoning is accurate, relevant, and sufficiently explained.",
        "Evaluate whether the response is complete enough to justify the score awarded.",
    ]


def _extract_prompt_scoring_rules(marking_guidance: str, max_mark: float | None) -> list[str]:
    rules: list[str] = []
    guidance_text = marking_guidance.lower()
    if max_mark is not None:
        rules.append(f"Use the full score range from 0 to {_format_number(max_mark)}.")
    if any(band in guidance_text for band in ("first", "2:1", "2:2", "third/pass", "missing/unfinished", "fail")):
        rules.append("Choose the quality band first, then place the score within that band using the evidence in this section.")
    if any(term in guidance_text for term in ("partial credit", "algebra", "equation", "deriv", "method", "interpretation")):
        rules.append("Award partial credit for correct setup, intermediate reasoning, and interpretation even when the final answer is incomplete.")
    if any(term in guidance_text for term in ("reference", "references", "page number", "page numbers", "quote", "quotes")):
        rules.append("Reward accurate use of specific references or page numbers when the task asks for them.")
    if any(term in guidance_text for term in ("rewrite", "own tone", "own words", "correct any mistakes", "edit the output")):
        rules.append("Reward correction of mistakes and rewriting in the student's own justified form when the task asks for it.")
    return rules


def _task_goal_from_task_type(task_type: str, question_text: str) -> str:
    if task_type == "critique_and_revision":
        return "generate, audit, and correct an AI-produced answer using supporting sources"
    if task_type == "model_specification":
        return "state the required formal specification and explain what each term means in context"
    if task_type == "deterministic_derivation":
        return "derive the required conditions and justify why they produce the stated scenario"
    if task_type == "explanation_interpretation":
        return "explain the model logic, parameter effects, or interpretation required by the question"
    if task_type == "welfare_reasoning":
        return "explain the welfare implications, relevant conditions, and effects on the affected parties"
    if task_type == "evaluative_discussion":
        return "make a justified evaluative judgement using the model or course concepts"
    if task_type == "prediction_generation":
        return "state the model's testable predictions clearly and correctly"
    if task_type == "measurement_and_data_design":
        return "show how the theoretical variables could be measured and linked to real data sources"
    if task_type == "regression_specification":
        return "write the regression equation, define its terms, and state the testable hypothesis"
    if task_type == "causal_identification":
        return "explain the endogeneity risks and propose feasible remedies"
    if task_type == "synthesis_across_sources":
        return "use theory and evidence to evaluate source claims and explain what the model misses"
    normalized = " ".join(question_text.split()).strip()
    if not normalized:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0].strip()
    return first_sentence[:220].rstrip()


def build_structure_messages(script_text: str, filename: str) -> list[dict[str, str]]:
    return build_structure_messages_with_guidance(script_text, filename, "")


def build_structure_messages_with_guidance(script_text: str, filename: str, structure_guidance: str) -> list[dict[str, str]]:
    system = (
        "You are detecting the assessment structure of a student's submission. "
        "Identify the substantive questions, parts, or sections the student is answering. "
        "Do not grade the work."
    )
    guidance_block = ""
    if structure_guidance.strip():
        guidance_block = (
            "STRUCTURE HINTS FROM MARKING DOCUMENTS:\n"
            f"{structure_guidance.strip()}\n\n"
        )
    user = (
        f"STUDENT FILE NAME: {filename}\n\n"
        f"{guidance_block}"
        "FULL STUDENT ANSWER:\n"
        f"\"\"\"{script_text}\"\"\"\n\n"
        "Return only one JSON object with exactly this structure:\n"
        '{ "sections": [ { "label": string, "focus_hint": string, "anchor_text": string } ] }\n\n'
        "Rules:\n"
        "1. Use the structure hints from marking documents only when they contain useful section labels, marks, or weights.\n"
        "2. If the structure hints are vague or irrelevant, rely on the student's submission instead.\n"
        "3. Choose the finest grading granularity that is clearly supported by both the marking documents and the student's submission.\n"
        "4. If subquestions are clearly separated and should be graded separately, return them individually; otherwise keep the coarser part.\n"
        "5. If you are not confident that a finer split is correct, prefer the coarser structure.\n"
        "6. label should be short, such as 'Question 1', 'Part 2', or 'Part 2 Q3'.\n"
        "7. focus_hint should briefly say what content belongs to that section.\n"
        "8. anchor_text must be a short exact quote copied verbatim from near the start of that section in the student submission.\n"
        "9. If the script does not clearly separate sections, return one section labelled 'Whole Submission'.\n"
        "10. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_subpart_structure_messages(parent_part: SubmissionPart, filename: str, child_specs: list[SubmissionPart]) -> list[dict[str, str]]:
    child_labels = ", ".join(
        f"{child.label} ({_format_number(child.max_mark) if child.max_mark is not None else '?'})"
        for child in child_specs
    )
    system = (
        "You are refining the grading structure inside one section of a student's submission. "
        "Only split this section into smaller grading units when the split is clearly supported by the section text. "
        "If you are not confident, keep the original parent section as one unit."
    )
    user = (
        f"STUDENT FILE NAME: {filename}\n"
        f"PARENT SECTION: {parent_part.label}\n"
        f"EXPECTED CHILD UNITS FROM MARKING DOCUMENTS: {child_labels}\n\n"
        "PARENT SECTION TEXT:\n"
        f"\"\"\"{parent_part.section_text}\"\"\"\n\n"
        "Return only one JSON object with exactly this structure:\n"
        '{ "sections": [ { "label": string, "focus_hint": string, "anchor_text": string } ] }\n\n'
        "Rules:\n"
        "1. Use child labels only when the student's section text clearly separates them.\n"
        "2. If the split is unclear, return one section with the parent label.\n"
        "3. anchor_text must be a short exact quote copied verbatim from near the start of that child section.\n"
        "4. Do not invent labels beyond the parent and expected child units.\n"
        "5. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_moderation_messages(
    script_text: str,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    context: MarkingContext,
    filename: str,
) -> list[dict[str, str]]:
    context_text = _build_context_text(context)
    payload = []
    for part, analysis in zip(parts, part_analyses):
        payload.append(
            {
                "section_label": part.label,
                "section_max_mark": part.max_mark,
                "current_provisional_score": analysis.get("provisional_score"),
                "current_provisional_score_0_to_100": analysis.get("provisional_score_0_to_100"),
                "strengths": analysis.get("strengths", []),
                "weaknesses": analysis.get("weaknesses", []),
                "evidence": analysis.get("evidence", []),
                "coverage_comment": analysis.get("coverage_comment", ""),
            }
        )
    analyses_text = json.dumps(payload, ensure_ascii=True, indent=2)
    system = (
        "You are moderating section-by-section marks within a single student's submission. "
        "Compare the sections against each other for internal consistency. "
        "Adjust a section only when the current score is clearly too harsh or too generous relative to the evidence and the other sections. "
        "The final overall mark will be computed separately by arithmetic, so you must only return section-level adjustments."
    )
    user = (
        f"{context_text}\n\n"
        f"STUDENT FILE NAME: {filename}\n\n"
        "FULL STUDENT ANSWER:\n"
        f"\"\"\"{script_text}\"\"\"\n\n"
        "CURRENT SECTION ANALYSES:\n"
        f"{analyses_text}\n\n"
        "Return only one JSON object with exactly this structure:\n"
        '{ "adjusted_sections": [ { "section_label": string, "adjusted_provisional_score": number|null, "adjusted_provisional_score_0_to_100": number|null, "rationale": string } ] }\n\n'
        "Rules:\n"
        "1. Include every section exactly once.\n"
        "2. Use adjusted_provisional_score only for sections that have an explicit max mark.\n"
        "3. Use adjusted_provisional_score_0_to_100 only when the section is on a percentage scale.\n"
        "4. Keep changes bounded and evidence-based; do not rewrite the whole assessment.\n"
        "5. If a current section score already looks internally consistent, keep it unchanged.\n"
        "6. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_synthesis_messages(
    script_text: str,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    context: MarkingContext,
    filename: str,
) -> list[dict[str, str]]:
    max_mark_text = _format_number(context.max_mark)
    part_labels = ", ".join(part.label for part in parts)
    analyses_text = json.dumps(part_analyses, ensure_ascii=True, indent=2)
    context_text = _build_context_text(context)
    part_weights_text = ", ".join(
        f"{part.label}={_format_number(part.max_mark)}" for part in parts if part.max_mark is not None
    )
    system = (
        "You are a fair and consistent university examiner completing the final synthesis stage. "
        "Use the supplied marking documents, the full student submission, and the per-section analyses. "
        "You must reward and penalize consistently across sections."
    )
    user = (
        "You are synthesizing a final mark from section-by-section analysis.\n\n"
        f"{context_text}\n\n"
        f"STUDENT FILE NAME: {filename}\n"
        f"DETECTED SECTIONS: {part_labels}\n\n"
        f"{'SECTION MARKS: ' + part_weights_text + chr(10) + chr(10) if part_weights_text else ''}"
        "FULL STUDENT ANSWER:\n"
        f"\"\"\"{script_text}\"\"\"\n\n"
        "SECTION ANALYSES:\n"
        f"{analyses_text}\n\n"
        "Return only one JSON object with exactly these keys:\n"
        '- "overall_feedback": string\n'
        '- "covered_parts": array of strings\n'
        '- "strengths": array of strings\n'
        '- "weaknesses": array of strings\n\n'
        "Rules:\n"
        f"1. The overall maximum mark is {max_mark_text}, but do not return any mark fields.\n"
        "2. Use the section analyses and section marks to support feedback; do not ignore a weaker section simply because other sections are strong.\n"
        "3. overall_feedback must be at least 140 words.\n"
        "4. overall_feedback must explicitly mention every detected section label.\n"
        "5. covered_parts must list every detected section label.\n"
        "6. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_calibration_messages(
    assessment_name: str,
    context: MarkingContext,
    model_label: str,
    student_summaries: list[dict[str, Any]],
) -> list[dict[str, str]]:
    max_mark_text = _format_number(context.max_mark)
    context_text = _build_context_text(context)
    summaries_text = json.dumps(student_summaries, ensure_ascii=True, indent=2)
    system = (
        "You are calibrating marks across a cohort after provisional grading. "
        "Your job is to improve consistency across students answering the same assessment. "
        "Do not invent evidence beyond the supplied summaries."
    )
    user = (
        f"ASSESSMENT: {assessment_name}\n"
        f"MODEL LABEL: {model_label}\n\n"
        f"{context_text}\n\n"
        "PROVISIONAL STUDENT SUMMARIES:\n"
        f"{summaries_text}\n\n"
        "Return only one JSON object with exactly this structure:\n"
        '{ "calibrated_results": [ { "filename": string, "adjusted_mark": number, "rationale": string } ] }\n\n'
        "Rules:\n"
        f"1. Every adjusted_mark must be between 0 and {max_mark_text}.\n"
        "2. Include every filename exactly once.\n"
        "3. Keep changes modest and evidence-based.\n"
        "4. Use the relative quality signals in the summaries to improve consistency across students and sections.\n"
        "5. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_verification_messages(
    script_text: str,
    context: MarkingContext,
    filename: str,
    grading_result: dict[str, Any],
) -> list[dict[str, str]]:
    context_text = _build_context_text(context)
    grading_summary = json.dumps(
        {
            "total_mark": grading_result.get("total_mark"),
            "max_mark": grading_result.get("max_mark"),
            "covered_parts": grading_result.get("covered_parts", []),
            "strengths": grading_result.get("strengths", []),
            "weaknesses": grading_result.get("weaknesses", []),
            "overall_feedback": grading_result.get("overall_feedback", ""),
        },
        ensure_ascii=True,
        indent=2,
    )
    system = (
        "You are verifying an existing assessment grade. "
        "Check whether the mark and feedback are plausible given the submission and the marking documents. "
        "Do not regrade from scratch unless needed to explain a discrepancy."
    )
    user = (
        f"{context_text}\n\n"
        f"STUDENT FILE NAME: {filename}\n\n"
        "FULL STUDENT ANSWER:\n"
        f"\"\"\"{script_text}\"\"\"\n\n"
        "PROPOSED GRADING RESULT:\n"
        f"{grading_summary}\n\n"
        "Return only one JSON object with exactly these keys:\n"
        '- "agreement": string\n'
        '- "confidence_0_to_100": number\n'
        '- "issues": array of strings\n'
        '- "recommendation": string\n\n'
        "Rules:\n"
        "1. agreement must be one of: agree, minor_concern, major_concern.\n"
        "2. issues should be concrete and brief.\n"
        "3. recommendation should say whether to accept, review, or regrade.\n"
        "4. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_regrade_messages(
    script_text: str,
    context: MarkingContext,
    filename: str,
    prior_result: dict[str, Any],
    verifier_result: dict[str, Any],
) -> list[dict[str, str]]:
    context_text = _build_context_text(context)
    prior_summary = json.dumps(
        {
            "total_mark": prior_result.get("total_mark"),
            "max_mark": prior_result.get("max_mark"),
            "covered_parts": prior_result.get("covered_parts", []),
            "strengths": prior_result.get("strengths", []),
            "weaknesses": prior_result.get("weaknesses", []),
            "overall_feedback": prior_result.get("overall_feedback", ""),
        },
        ensure_ascii=True,
        indent=2,
    )
    verifier_summary = json.dumps(verifier_result, ensure_ascii=True, indent=2)
    max_mark_text = _format_number(context.max_mark)
    system = (
        "You are revising a grading decision after verifier feedback. "
        "Reassess the mark and feedback carefully, using the verifier concerns only when they are valid. "
        "Do not defend the previous answer by reflex."
    )
    user = (
        f"{context_text}\n\n"
        f"STUDENT FILE NAME: {filename}\n\n"
        "FULL STUDENT ANSWER:\n"
        f"\"\"\"{script_text}\"\"\"\n\n"
        "PRIOR GRADING RESULT:\n"
        f"{prior_summary}\n\n"
        "VERIFIER FEEDBACK:\n"
        f"{verifier_summary}\n\n"
        "Return only one JSON object with exactly these keys:\n"
        '- "total_mark": number\n'
        '- "max_mark": number\n'
        '- "overall_feedback": string\n'
        '- "covered_parts": array of strings\n'
        '- "strengths": array of strings\n'
        '- "weaknesses": array of strings\n\n'
        "Rules:\n"
        f"1. Use max_mark = {max_mark_text}.\n"
        f"2. total_mark must be between 0 and {max_mark_text}.\n"
        "3. overall_feedback must be at least 140 words.\n"
        "4. Address verifier concerns when they are supported by the submission and marking documents.\n"
        "5. If verifier concerns are not supported, keep the stronger original judgement.\n"
        "6. Do not include markdown fences or any text outside the JSON object."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_ollama(
    script_text: str,
    context: MarkingContext,
    filename: str,
    model_name: str,
    rubric_verifier_model_name: str | None = None,
    structure_script_text: str | None = None,
    prepared_assessment_map: PreparedAssessmentMap | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    return call_ollama_comparative_first(
        script_text=script_text,
        context=context,
        filename=filename,
        model_name=model_name,
        rubric_verifier_model_name=rubric_verifier_model_name,
        structure_script_text=structure_script_text,
        prepared_assessment_map=prepared_assessment_map,
        ollama_url=ollama_url,
    )


def call_ollama_comparative_first(
    script_text: str,
    context: MarkingContext,
    filename: str,
    model_name: str,
    rubric_verifier_model_name: str | None = None,
    structure_script_text: str | None = None,
    prepared_assessment_map: PreparedAssessmentMap | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    timings: dict[str, float] = {
        "assessment_prepare": 0.0,
        "structure_detect": 0.0,
        "part_refine": 0.0,
        "part_analysis": 0.0,
        "moderation": 0.0,
        "finalize": 0.0,
    }
    script_text = script_text.strip()
    if not script_text:
        raise ValueError(
            "No text could be extracted from the submission. "
            "OCR is not implemented yet, so use a text-based PDF, DOCX, or CSV answer."
        )
    use_alternate_structure_text = bool((structure_script_text or "").strip()) and bool(extract_expected_parts_from_context(context))
    structure_input_text = (structure_script_text or "").strip() if use_alternate_structure_text else script_text

    if prepared_assessment_map is None:
        stage_started = time.perf_counter()
        prepared_assessment_map = prepare_assessment_map(
            context=context,
            verifier_model_name=rubric_verifier_model_name,
            ollama_url=ollama_url,
        )
        timings["assessment_prepare"] = round(time.perf_counter() - stage_started, 2)
    else:
        timings["assessment_prepare"] = round(prepared_assessment_map.preparation_latency_seconds, 2)
    assessment_map = prepared_assessment_map.prepared_map
    total_started = time.perf_counter()
    stage_started = time.perf_counter()
    detected_parts = detect_submission_parts(structure_input_text, filename, model_name, context=context, ollama_url=ollama_url)
    timings["structure_detect"] = round(time.perf_counter() - stage_started, 2)
    stage_started = time.perf_counter()
    parts = segment_submission_parts(script_text, detected_parts)
    parts = refine_submission_granularity(parts, context, filename, model_name, ollama_url=ollama_url)
    parts = apply_assessment_map_to_submission_parts(parts, assessment_map)
    parts = apply_prepared_artifact_to_submission_parts(parts, prepared_assessment_map)
    timings["part_refine"] = round(time.perf_counter() - stage_started, 2)
    diagnostics = build_submission_diagnostics(script_text, parts)

    part_analyses = []
    stage_started = time.perf_counter()
    for part in parts:
        if not part.section_text.strip():
            part_analyses.append(build_missing_part_analysis(part))
            continue
        part_analyses.append(
            _run_part_analysis_with_retry(
                part=part,
                context=context,
                filename=filename,
                model_name=model_name,
                ollama_url=ollama_url,
            )
        )
    timings["part_analysis"] = round(time.perf_counter() - stage_started, 2)

    if len(part_analyses) > 1:
        stage_started = time.perf_counter()
        part_analyses = moderate_linked_part_analyses(
            script_text=script_text,
            context=context,
            filename=filename,
            parts=parts,
            part_analyses=part_analyses,
            assessment_map=assessment_map,
            model_name=model_name,
            ollama_url=ollama_url,
        )
        timings["moderation"] = round(time.perf_counter() - stage_started, 2)

    if part_analyses:
        stage_started = time.perf_counter()
        result = build_local_final_result(
            context=context,
            diagnostics=diagnostics,
            parts=parts,
            part_analyses=part_analyses,
        )
        timings["finalize"] = round(time.perf_counter() - stage_started, 2)
        return _attach_latency_metrics(result, timings, total_started, parts)

    synthesis_messages = build_synthesis_messages(script_text, parts, part_analyses, context, filename)
    stage_started = time.perf_counter()
    result = _run_final_result_with_retry(
        model_name=model_name,
        messages=synthesis_messages,
        expected_max_mark=context.max_mark,
        diagnostics=diagnostics,
        parts=parts,
        part_analyses=part_analyses,
        ollama_url=ollama_url,
    )
    timings["finalize"] = round(time.perf_counter() - stage_started, 2)
    return _attach_latency_metrics(result, timings, total_started, parts)


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


def detect_submission_parts(
    script_text: str,
    filename: str,
    model_name: str,
    context: MarkingContext | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[SubmissionPart]:
    context_fast_path = _detect_submission_parts_from_context(script_text, context)
    if context_fast_path is not None:
        return context_fast_path
    structure_guidance = extract_structure_guidance(context) if context is not None else ""
    data = _call_ollama_json(
        model_name=model_name,
        messages=build_structure_messages_with_guidance(script_text, filename, structure_guidance),
        ollama_url=ollama_url,
    )
    return reconcile_detected_parts(normalize_detected_parts(data), context)


def describe_structure_detection_mode(
    script_text: str,
    context: MarkingContext | None,
) -> dict[str, Any]:
    context_fast_path = _detect_submission_parts_from_context(script_text, context)
    expected_parts = extract_expected_parts_from_context(context) if context is not None else []
    if context_fast_path is not None:
        return {
            "mode": "context_fast_path",
            "expected_part_count": len(expected_parts),
            "detected_part_count": len(context_fast_path),
        }
    return {
        "mode": "model_call",
        "expected_part_count": len(expected_parts),
        "detected_part_count": None,
    }


def _detect_submission_parts_from_context(
    script_text: str,
    context: MarkingContext | None,
) -> list[SubmissionPart] | None:
    if context is None:
        return None
    expected_parts = extract_expected_parts_from_context(context)
    if not expected_parts:
        return None
    if len(expected_parts) == 1:
        return expected_parts

    normalized_text = script_text.replace("\r\n", "\n")
    anchored_count = sum(1 for part in expected_parts if _find_part_anchor_position(normalized_text, part) != -1)
    if anchored_count >= max(2, len(expected_parts) // 2):
        return expected_parts
    return None


def normalize_detected_parts(data: dict[str, Any]) -> list[SubmissionPart]:
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        return [SubmissionPart(label="Whole Submission", focus_hint="Assess the full script as one piece of work.", anchor_text="")]

    parts: list[SubmissionPart] = []
    seen_labels: set[str] = set()
    for item in sections:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label or _is_placeholder_section_label(label):
            continue
        focus_hint = str(item.get("focus_hint", "")).strip()
        anchor_text = str(item.get("anchor_text", "")).strip()
        if label in seen_labels:
            continue
        seen_labels.add(label)
        parts.append(SubmissionPart(label=label, focus_hint=focus_hint, anchor_text=anchor_text))

    if not parts:
        return [SubmissionPart(label="Whole Submission", focus_hint="Assess the full script as one piece of work.", anchor_text="")]
    return parts


def reconcile_detected_parts(detected_parts: list[SubmissionPart], context: MarkingContext | None) -> list[SubmissionPart]:
    if context is None:
        return detected_parts

    expected_parts = extract_expected_parts_from_context(context)
    if not expected_parts:
        return detected_parts

    expected_by_key = {_normalize_label_key(part.label): part for part in expected_parts}
    detected_by_key = {
        _normalize_label_key(part.label): part
        for part in detected_parts
        if _normalize_label_key(part.label) in expected_by_key
    }
    enriched = []
    for expected in expected_parts:
        part = detected_by_key.get(_normalize_label_key(expected.label), expected)
        enriched.append(
            SubmissionPart(
                label=expected.label,
                focus_hint=part.focus_hint or expected.focus_hint,
                anchor_text=part.anchor_text or expected.anchor_text,
                section_text=part.section_text,
                max_mark=expected.max_mark,
                marking_guidance=expected.marking_guidance,
                question_text_exact=expected.question_text_exact,
                task_type=expected.task_type,
            )
        )
    return enriched


def segment_submission_parts(script_text: str, detected_parts: list[SubmissionPart]) -> list[SubmissionPart]:
    if len(detected_parts) == 1 and detected_parts[0].label == "Whole Submission":
        return [SubmissionPart(label="Whole Submission", focus_hint=detected_parts[0].focus_hint, anchor_text="", section_text=script_text.strip(), max_mark=detected_parts[0].max_mark, marking_guidance=detected_parts[0].marking_guidance, question_text_exact=detected_parts[0].question_text_exact, task_type=detected_parts[0].task_type)]

    normalized_text = script_text.replace("\r\n", "\n")
    anchors: list[tuple[int, SubmissionPart]] = []
    used_positions: set[int] = set()
    for part in detected_parts:
        position = _find_part_anchor_position(normalized_text, part)
        if position == -1 or position in used_positions:
            continue
        used_positions.add(position)
        anchors.append((position, part))

    if len(anchors) < max(1, len(detected_parts) // 2):
        return [
            SubmissionPart(
                label=part.label,
                focus_hint=part.focus_hint,
                anchor_text=part.anchor_text,
                section_text=script_text.strip(),
                max_mark=part.max_mark,
                marking_guidance=part.marking_guidance,
                question_text_exact=part.question_text_exact,
                task_type=part.task_type,
            )
            for part in detected_parts
        ]

    anchors.sort(key=lambda item: item[0])
    segmented: list[SubmissionPart] = []
    for index, (start, part) in enumerate(anchors):
        end = anchors[index + 1][0] if index + 1 < len(anchors) else len(normalized_text)
        section_text = normalized_text[start:end].strip()
        segmented.append(
            SubmissionPart(
                label=part.label,
                focus_hint=part.focus_hint,
                anchor_text=part.anchor_text,
                section_text=section_text,
                max_mark=part.max_mark,
                marking_guidance=part.marking_guidance,
                question_text_exact=part.question_text_exact,
                task_type=part.task_type,
            )
        )
    return segmented


def refine_submission_granularity(
    segmented_parts: list[SubmissionPart],
    context: MarkingContext | None,
    filename: str,
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[SubmissionPart]:
    if context is None or not segmented_parts:
        return segmented_parts

    child_map = extract_expected_subparts_from_context(context)
    if not child_map:
        return segmented_parts

    refined: list[SubmissionPart] = []
    for part in segmented_parts:
        child_specs = child_map.get(_normalize_label_key(part.label), [])
        if len(child_specs) < 2:
            refined.append(part)
            continue
        if not part.section_text.strip():
            refined.extend(_resolve_expected_child_parts(parent_part=part, child_parts=[] , child_specs=child_specs))
            continue
        segmented_children = _segment_subparts_within_section(part, child_specs)
        if segmented_children is None:
            segmented_children = _detect_subparts_with_model(part, filename, child_specs, model_name, ollama_url)
        if segmented_children is None:
            refined.extend(_resolve_expected_child_parts(parent_part=part, child_parts=[], child_specs=child_specs))
            continue
        refined.extend(_resolve_expected_child_parts(parent_part=part, child_parts=segmented_children, child_specs=child_specs))
    return refined


def extract_structure_guidance(context: MarkingContext) -> str:
    source_text = "\n\n".join(
        text.strip()
        for text in (context.rubric_text, context.brief_text, context.marking_scheme_text)
        if text and text.strip()
    )
    if not source_text:
        return ""

    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    useful_lines: list[str] = []
    patterns = (
        r"\b(question|part|section)\s+[a-z0-9ivx]+\b",
        r"\b\d+\s*(marks?|%)\b",
        r"\bweight(?:ing)?\b",
        r"\bout of\s+\d+\b",
    )
    for line in lines:
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
            useful_lines.append(line)
        if len(useful_lines) >= 12:
            break

    return "\n".join(useful_lines)


def extract_expected_parts_from_context(context: MarkingContext) -> list[SubmissionPart]:
    source_text = "\n\n".join(
        text.strip()
        for text in (context.rubric_text, context.brief_text, context.marking_scheme_text)
        if text and text.strip()
    )
    if not source_text:
        return []

    parts: list[SubmissionPart] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"^(?P<kind>question|part|section)\s+(?P<id>\d+|[ivx]+|[a-d])\b(?:\s*[:.)-]|\s*\((?P<marks>\d+(?:\.\d+)?)\s*marks?\)|$)",
        flags=re.IGNORECASE,
    )
    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        label = f"{match.group('kind').title()} {match.group('id')}"
        key = _normalize_label_key(label)
        if key in seen:
            continue
        seen.add(key)
        marks = float(match.group("marks")) if match.group("marks") else None
        guidance_lines = [line.strip()]
        for extra in lines[index + 1 :]:
            if pattern.search(extra):
                break
            if extra.lower().startswith("answer:"):
                break
            if len(guidance_lines) >= 18:
                break
            if extra.isdigit():
                continue
            guidance_lines.append(extra)
        question_text_exact = "\n".join(guidance_lines).strip()
        parts.append(
            SubmissionPart(
                label=label,
                focus_hint=line.strip(),
                anchor_text=label,
                max_mark=marks,
                marking_guidance="\n".join(guidance_lines),
                question_text_exact=question_text_exact,
            )
        )
    return parts


def extract_expected_subparts_from_context(context: MarkingContext) -> dict[str, list[SubmissionPart]]:
    source_text = "\n\n".join(
        text.strip()
        for text in (context.rubric_text, context.brief_text, context.marking_scheme_text)
        if text and text.strip()
    )
    if not source_text:
        return {}

    top_level_pattern = re.compile(
        r"^(?P<kind>question|part|section)\s+(?P<id>\d+|[ivx]+|[a-d])\b(?:\s*[:.)-]|\s*\((?P<marks>\d+(?:\.\d+)?)\s*marks?\)|$)",
        flags=re.IGNORECASE,
    )
    subpart_pattern = re.compile(
        r"^(?P<id>\d+)\.\s*\[(?P<marks>\d+(?:\.\d+)?)\s*marks?\]",
        flags=re.IGNORECASE,
    )

    lines = [line.strip() for line in source_text.splitlines() if line.strip()]
    child_map: dict[str, list[SubmissionPart]] = {}
    current_parent: str | None = None
    current_children: list[SubmissionPart] = []

    def flush_children() -> None:
        nonlocal current_parent, current_children
        if current_parent and len(current_children) >= 2:
            child_map[current_parent] = list(current_children)
        current_children = []

    for index, line in enumerate(lines):
        top_match = top_level_pattern.search(line)
        if top_match:
            flush_children()
            current_parent = _normalize_label_key(f"{top_match.group('kind').title()} {top_match.group('id')}")
            continue
        if current_parent is None:
            continue
        sub_match = subpart_pattern.search(line)
        if not sub_match:
            continue
        question_id = sub_match.group("id")
        guidance_lines = [line]
        for extra in lines[index + 1 :]:
            if top_level_pattern.search(extra) or subpart_pattern.search(extra):
                break
            if extra.lower().startswith("answer:"):
                break
            guidance_lines.append(extra)
            if len(guidance_lines) >= 6:
                break
        question_text_exact = "\n".join(guidance_lines).strip()
        current_children.append(
            SubmissionPart(
                label=f"{_restore_label_from_key(current_parent)} Q{question_id}",
                focus_hint=line,
                anchor_text=f"Q{question_id}",
                max_mark=float(sub_match.group("marks")),
                marking_guidance=question_text_exact,
                question_text_exact=question_text_exact,
            )
        )

    flush_children()
    return child_map


def build_submission_diagnostics(script_text: str, parts: list[SubmissionPart] | None = None) -> SubmissionDiagnostics:
    words = script_text.split()
    extracted_word_count = len(words)
    resolved_parts = parts or [SubmissionPart(label="Whole Submission", focus_hint="Assess the full script as one piece of work.")]
    possible_extraction_issue = bool(re.search(r"(.)\1{8,}", script_text)) or extracted_word_count < 80
    return SubmissionDiagnostics(
        extracted_word_count=extracted_word_count,
        detected_part_count=len(resolved_parts),
        detected_part_labels=tuple(part.label for part in resolved_parts),
        low_text=extracted_word_count < 120,
        possible_extraction_issue=possible_extraction_issue,
    )


def split_submission_into_parts(script_text: str) -> list[SubmissionPart]:
    return [SubmissionPart(label="Whole Submission", focus_hint="Assess the full script as one piece of work.")]


def build_assessment_map_cache_key(
    context: MarkingContext,
    verifier_model_name: str | None = None,
) -> str:
    payload = {
        "rubric_text": context.rubric_text,
        "brief_text": context.brief_text,
        "marking_scheme_text": context.marking_scheme_text,
        "graded_sample_text": context.graded_sample_text,
        "other_context_text": context.other_context_text,
        "max_mark": context.max_mark,
        "verifier_model_name": verifier_model_name or "",
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def clear_prepared_assessment_map_cache() -> None:
    _PREPARED_ASSESSMENT_CACHE.clear()


def _is_verifier_failure_issue(issue: str) -> bool:
    return issue.strip().startswith("rubric_verifier_failed:")


def _collect_preparation_issues(assessment_map: AssessmentMap) -> tuple[str, ...]:
    issues: list[str] = []
    for unit in assessment_map.units:
        issues.extend(issue for issue in unit.rubric_issues if issue.strip())
    return tuple(issues)


def _compute_preparation_confidence(assessment_map: AssessmentMap) -> float | None:
    unit_confidences = [
        unit.rubric_confidence_0_to_100
        for unit in assessment_map.units
        if unit.rubric_confidence_0_to_100 is not None
    ]
    if not unit_confidences:
        return None
    return round(sum(unit_confidences) / len(unit_confidences), 2)


def _unit_refinement_score(unit: AssessmentUnit) -> float:
    confidence = unit.rubric_confidence_0_to_100 if unit.rubric_confidence_0_to_100 is not None else 0.0
    issue_penalty = float(len(unit.rubric_issues))
    failure_penalty = 25.0 if any(_is_verifier_failure_issue(issue) for issue in unit.rubric_issues) else 0.0
    return round(confidence - issue_penalty - failure_penalty, 2)


def _summarize_unit_refinement(previous: AssessmentUnit, candidate: AssessmentUnit, attempt_number: int) -> str:
    previous_confidence = previous.rubric_confidence_0_to_100 if previous.rubric_confidence_0_to_100 is not None else 0.0
    candidate_confidence = candidate.rubric_confidence_0_to_100 if candidate.rubric_confidence_0_to_100 is not None else 0.0
    previous_issue_count = len(previous.rubric_issues)
    candidate_issue_count = len(candidate.rubric_issues)
    changed_text = previous.rubric_text.strip() != candidate.rubric_text.strip()
    text_change = "text_changed" if changed_text else "text_unchanged"
    return (
        f"attempt_{attempt_number}: confidence {previous_confidence:.2f}->{candidate_confidence:.2f}, "
        f"issues {previous_issue_count}->{candidate_issue_count}, {text_change}"
    )


def _is_acceptable_prepared_map(assessment_map: AssessmentMap) -> bool:
    confidences = [
        unit.rubric_confidence_0_to_100
        for unit in assessment_map.units
        if unit.rubric_confidence_0_to_100 is not None
    ]
    if not confidences:
        return True
    if any(any(_is_verifier_failure_issue(issue) for issue in unit.rubric_issues) for unit in assessment_map.units):
        return False
    return min(confidences) >= MIN_ACCEPTABLE_PREPARATION_CONFIDENCE


def _build_prepared_assessment_units(
    assessment_map: AssessmentMap,
    verifier_attempt_count: int,
    accepted_refinements: dict[str, list[str]],
    rejected_refinements: dict[str, list[str]],
) -> tuple[PreparedAssessmentUnit, ...]:
    prepared_units: list[PreparedAssessmentUnit] = []
    for unit in assessment_map.units:
        unit_key = _normalize_label_key(unit.label)
        prepared_units.append(
            PreparedAssessmentUnit(
                label=unit.label,
                compact_criteria=tuple(_extract_classification_criteria(unit)),
                verifier_confidence_0_to_100=unit.rubric_confidence_0_to_100,
                verifier_issues=tuple(issue for issue in unit.rubric_issues if issue.strip()),
                verifier_attempt_count=verifier_attempt_count,
                verifier_accepted_refinements=tuple(accepted_refinements.get(unit_key, [])),
                verifier_rejected_refinements=tuple(rejected_refinements.get(unit_key, [])),
            )
        )
    return tuple(prepared_units)


def _merge_refined_assessment_maps(
    current_map: AssessmentMap,
    candidate_map: AssessmentMap,
    attempt_number: int,
    accepted_refinements: dict[str, list[str]],
    rejected_refinements: dict[str, list[str]],
) -> AssessmentMap:
    merged_units: list[AssessmentUnit] = []
    for current_unit, candidate_unit in zip(current_map.units, candidate_map.units):
        unit_key = _normalize_label_key(current_unit.label)
        changed = (
            current_unit.rubric_text.strip() != candidate_unit.rubric_text.strip()
            or current_unit.rubric_confidence_0_to_100 != candidate_unit.rubric_confidence_0_to_100
            or current_unit.rubric_issues != candidate_unit.rubric_issues
        )
        if _unit_refinement_score(candidate_unit) > _unit_refinement_score(current_unit):
            merged_units.append(candidate_unit)
            if changed:
                accepted_refinements[unit_key].append(
                    _summarize_unit_refinement(current_unit, candidate_unit, attempt_number)
                )
            continue
        merged_units.append(current_unit)
        if changed:
            rejected_refinements[unit_key].append(
                _summarize_unit_refinement(current_unit, candidate_unit, attempt_number)
            )
    return replace(current_map, units=tuple(merged_units))


def prepare_assessment_map(
    context: MarkingContext,
    verifier_model_name: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> PreparedAssessmentMap:
    cache_key = build_assessment_map_cache_key(
        context=context,
        verifier_model_name=verifier_model_name,
    )
    cached = _PREPARED_ASSESSMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    original_map = build_assessment_map(context)
    prepared_map = original_map
    verification_applied = False
    verifier_attempt_count = 0
    accepted_refinements: dict[str, list[str]] = defaultdict(list)
    rejected_refinements: dict[str, list[str]] = defaultdict(list)

    if verifier_model_name:
        verification_applied = True
        prepared_map = original_map
        for attempt_number in range(1, MAX_PREPARED_MAP_VERIFIER_ATTEMPTS + 1):
            verifier_attempt_count = attempt_number
            candidate_map = verify_assessment_rubrics(
                assessment_map=prepared_map,
                context=context,
                verifier_model_name=verifier_model_name,
                ollama_url=ollama_url,
            )
            if attempt_number == 1:
                prepared_map = candidate_map
            else:
                prepared_map = _merge_refined_assessment_maps(
                    current_map=prepared_map,
                    candidate_map=candidate_map,
                    attempt_number=attempt_number,
                    accepted_refinements=accepted_refinements,
                    rejected_refinements=rejected_refinements,
                )
            if _is_acceptable_prepared_map(prepared_map):
                break
    preparation_confidence = _compute_preparation_confidence(prepared_map)
    preparation_issues = _collect_preparation_issues(prepared_map)
    preparation_latency_seconds = round(time.perf_counter() - started, 2)

    prepared = PreparedAssessmentMap(
        cache_key=cache_key,
        original_map=original_map,
        prepared_map=prepared_map,
        prepared_units=_build_prepared_assessment_units(
            assessment_map=prepared_map,
            verifier_attempt_count=verifier_attempt_count,
            accepted_refinements=accepted_refinements,
            rejected_refinements=rejected_refinements,
        ),
        verifier_model_name=verifier_model_name,
        verification_applied=verification_applied,
        verifier_attempt_count=verifier_attempt_count,
        preparation_issues=preparation_issues,
        preparation_confidence_0_to_100=preparation_confidence,
        preparation_latency_seconds=preparation_latency_seconds,
    )
    _PREPARED_ASSESSMENT_CACHE[cache_key] = prepared
    return prepared


def build_assessment_map(context: MarkingContext) -> AssessmentMap:
    top_level_parts = extract_expected_parts_from_context(context)
    child_map = extract_expected_subparts_from_context(context)

    units: list[AssessmentUnit] = []
    if top_level_parts:
        for part in top_level_parts:
            children = child_map.get(_normalize_label_key(part.label), [])
            if children:
                for child in children:
                    question_text = child.question_text_exact or child.marking_guidance or part.question_text_exact or part.marking_guidance or part.focus_hint
                    task_type = infer_task_type(child.label, question_text)
                    mode = infer_grading_mode(child.label, child.marking_guidance or part.marking_guidance or part.focus_hint, task_type=task_type)
                    dependency = infer_dependency_group(child.label, child.marking_guidance, parent_label=part.label)
                    units.append(
                        AssessmentUnit(
                            label=child.label,
                            max_mark=child.max_mark,
                            parent_label=part.label,
                            grading_mode=mode,
                            task_type=task_type,
                            dependency_group=dependency,
                            question_text_exact=question_text,
                            marking_guidance=child.marking_guidance,
                            rubric_text=build_unit_rubric_text(
                                label=child.label,
                                max_mark=child.max_mark,
                                grading_mode=mode,
                                task_type=task_type,
                                question_text_exact=question_text,
                                marking_guidance=child.marking_guidance,
                            ),
                        )
                    )
            else:
                question_text = part.question_text_exact or part.marking_guidance or part.focus_hint
                task_type = infer_task_type(part.label, question_text)
                mode = infer_grading_mode(part.label, part.marking_guidance or part.focus_hint, task_type=task_type)
                dependency = infer_dependency_group(part.label, part.marking_guidance, parent_label=part.label)
                units.append(
                    AssessmentUnit(
                        label=part.label,
                        max_mark=part.max_mark,
                        parent_label="",
                        grading_mode=mode,
                        task_type=task_type,
                        dependency_group=dependency,
                        question_text_exact=question_text,
                        marking_guidance=part.marking_guidance,
                        rubric_text=build_unit_rubric_text(
                            label=part.label,
                            max_mark=part.max_mark,
                            grading_mode=mode,
                            task_type=task_type,
                            question_text_exact=question_text,
                            marking_guidance=part.marking_guidance,
                        ),
                    )
                )

    if not units:
        units.append(
            AssessmentUnit(
                label="Whole Submission",
                max_mark=context.max_mark,
                grading_mode="analytical",
                task_type="whole_submission_holistic",
                dependency_group="",
                question_text_exact="Assess the full submission as one unit.",
                marking_guidance="Assess the full submission as one unit.",
                rubric_text=build_unit_rubric_text(
                    label="Whole Submission",
                    max_mark=context.max_mark,
                    grading_mode="analytical",
                    task_type="whole_submission_holistic",
                    question_text_exact="Assess the full submission as one unit.",
                    marking_guidance="Assess the full submission as one unit.",
                ),
            )
        )

    scale_confidence = 95.0 if any(unit.max_mark is not None for unit in units) else 60.0
    return AssessmentMap(
        units=tuple(units),
        overall_max_mark=context.max_mark,
        scale_confidence_0_to_100=scale_confidence,
    )


def build_rubric_matrix_markdown(context: MarkingContext, assessment_name: str = "Assessment") -> str:
    assessment_map = build_assessment_map(context)
    lines = [
        f"# {assessment_name} Rubric Matrix",
        "",
        f"Overall maximum mark: {_format_number(context.max_mark)}",
        "",
        "## Policy",
        "",
        "- This rubric is a classification matrix, not a transcript of the marking scheme.",
        "- Classification happens at part or subpart level first; the final mark is deterministic arithmetic from those unit scores.",
        "- Analytical units use UK-style bands. Deterministic units use evidence bands tied to correctness and method.",
        "- Within-band placement must be justified by concrete evidence from the script, not by generic prose.",
        "",
    ]
    grouped_units: dict[str, list[AssessmentUnit]] = {}
    for unit in assessment_map.units:
        grouped_units.setdefault(unit.parent_label or unit.label, []).append(unit)

    for group_label, units in grouped_units.items():
        lines.extend([f"## {group_label}", ""])
        for unit in units:
            lines.extend(_render_unit_rubric_matrix(unit))
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_rubric_verification_messages(unit: AssessmentUnit, context: MarkingContext) -> list[dict[str, str]]:
    max_mark_text = _format_number(unit.max_mark) if unit.max_mark is not None else _format_number(context.max_mark)
    structure_context = extract_structure_guidance(context)
    system = (
        "You are verifying and enriching one grading rubric unit before marking begins. "
        "You must preserve the unit label, max mark, grading mode, and dependency structure. "
        "You may only improve the rubric prose so it is clearer, more discriminating, and better aligned with the assessment documents. "
        'Return JSON with keys "rubric_text", "issues", and "confidence_0_to_100".'
    )
    user = (
        f"Unit label: {unit.label}\n"
        f"Parent label: {unit.parent_label or 'None'}\n"
        f"Unit max mark: {max_mark_text}\n"
        f"Grading mode: {unit.grading_mode}\n"
        f"Dependency group: {unit.dependency_group or 'independent'}\n\n"
        f"Current rubric text:\n{unit.rubric_text or unit.marking_guidance or '(empty)'}\n\n"
        f"Structure and marks evidence from the assessment documents:\n{structure_context or '(none)'}\n\n"
        "Verification rules:\n"
        "1. Do not assign a score and do not write a final mark.\n"
        "2. Do not change the label, max mark, part structure, or dependency structure.\n"
        "3. Use the assessment documents only when they add useful information about structure, marks, weights, or question numbering.\n"
        "4. If the unit is deterministic, enforce full-range discrimination, partial credit, method marks, equation or algebra checks, interpretation checks, and avoid compressing marks into the middle.\n"
        "5. If the unit is analytical, enforce UK classification language and distinctions: First, 2:1, 2:2, Third/Pass, Fail, Missing/unfinished. Map those bands to the unit mark range and make the top band rare and evidence-based.\n"
        "6. If a rule is already good, keep it.\n"
        "7. Put any concerns or missing evidence in issues.\n"
        "8. confidence_0_to_100 must be a number from 0 to 100.\n"
        "9. Return only JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_verified_rubric(data: dict[str, Any], unit: AssessmentUnit) -> dict[str, Any]:
    rubric_text = data.get("rubric_text")
    if not isinstance(rubric_text, str) or not rubric_text.strip():
        raise ValueError(f"Verifier did not return rubric_text for {unit.label}: {data}")

    issues = data.get("issues", [])
    if issues is None:
        issues = []
    if not isinstance(issues, list) or any(not isinstance(item, str) or not item.strip() for item in issues):
        raise ValueError(f"Verifier returned invalid issues for {unit.label}: {data}")

    confidence = data.get("confidence_0_to_100")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"Verifier returned invalid confidence for {unit.label}: {data}")
    confidence_value = round(float(confidence), 2)
    if confidence_value < 0 or confidence_value > 100:
        raise ValueError(f"Verifier returned out-of-range confidence for {unit.label}: {data}")

    return {
        "rubric_text": rubric_text.strip(),
        "issues": tuple(item.strip() for item in issues if item.strip()),
        "confidence_0_to_100": confidence_value,
    }


def verify_assessment_rubrics(
    assessment_map: AssessmentMap,
    context: MarkingContext,
    verifier_model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> AssessmentMap:
    verified_units: list[AssessmentUnit] = []
    for unit in assessment_map.units:
        messages = build_rubric_verification_messages(unit, context)
        normalized: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                data = _call_ollama_json(
                    model_name=verifier_model_name,
                    messages=messages,
                    ollama_url=ollama_url,
                )
                normalized = normalize_verified_rubric(data, unit)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    messages = _build_rubric_verification_retry_messages(messages, unit, exc)
                    continue

        if normalized is None:
            fallback_issues = list(unit.rubric_issues)
            if last_error is not None:
                fallback_issues.append(f"rubric_verifier_failed: {last_error}")
            verified_units.append(
                replace(
                    unit,
                    rubric_confidence_0_to_100=0.0,
                    rubric_issues=tuple(fallback_issues),
                )
            )
            continue

        verified_units.append(
            replace(
                unit,
                rubric_text=normalized["rubric_text"],
                rubric_confidence_0_to_100=normalized["confidence_0_to_100"],
                rubric_issues=normalized["issues"],
            )
        )

    return replace(assessment_map, units=tuple(verified_units))


def infer_task_type(label: str, question_text: str) -> str:
    text = " ".join(f"{label}\n{question_text}".lower().split())
    if (
        any(term in text for term in ("generate", "draft", "produce"))
        and any(term in text for term in ("summary", "answer", "response"))
        and any(term in text for term in ("comment on whether", "review", "critique", "evaluate accuracy", "rewrite", "revise", "edit", "correct"))
    ):
        return "critique_and_revision"
    if (
        any(term in text for term in ("write out", "write down", "state", "specify", "formulate"))
        and any(term in text for term in ("payoff function", "utility function", "objective function", "formal model", "formal specification"))
    ):
        return "model_specification"
    if any(term in text for term in ("derive", "show that", "solve for", "prove that")) and any(term in text for term in ("condition", "equilibrium", "solution")):
        return "deterministic_derivation"
    if any(term in text for term in ("welfare", "well-being", "wellbeing", "efficiency")) and any(term in text for term in ("condition", "implication", "effect", "impact")):
        return "welfare_reasoning"
    if any(term in text for term in ("list", "state", "give", "identify")) and any(term in text for term in ("prediction", "predictions", "implication", "implications", "testable hypothesis")):
        return "prediction_generation"
    if (
        any(term in text for term in ("measure", "measured", "measurement", "operationalise", "operationalize", "indicator", "observable", "observed"))
        and any(term in text for term in ("data source", "data sources", "dataset", "datasets", "real world", "practice", "in practice"))
    ):
        return "measurement_and_data_design"
    if (
        any(term in text for term in ("regression equation", "empirical specification", "estimating equation", "econometric specification"))
        and any(term in text for term in ("write down", "state", "specify", "formulate"))
    ):
        return "regression_specification"
    if any(term in text for term in ("endogeneity", "omitted variable", "reverse causality", "selection bias", "simultaneity", "confounding")):
        return "causal_identification"
    if (
        any(term in text for term in ("source", "sources", "article", "articles", "video", "videos", "text", "texts", "evidence"))
        and any(term in text for term in ("claim", "claims", "argument", "arguments", "compare", "contrast", "synthes", "missing from", "omission", "limitations", "shed light on"))
    ):
        return "synthesis_across_sources"
    if any(phrase in text for phrase in ("explain how changing", "explain carefully why", "discuss the intuition", "explain how each term", "explain how each variable", "interpret the coefficient", "interpret the result", "explain the mechanism")):
        return "explanation_interpretation"
    if any(term in text for term in ("discuss", "evaluate", "comment on whether", "whether you think", "critically assess")):
        return "evaluative_discussion"
    if any(term in text for term in ("summary", "overview", "comment", "assess")):
        return "evaluative_discussion"
    return "deterministic_derivation"


def _matches_all(text: str, *terms: str) -> bool:
    return all(term in text for term in terms)


def infer_grading_mode(label: str, marking_guidance: str, task_type: str = "") -> str:
    if task_type in {
        "critique_and_revision",
        "evaluative_discussion",
        "synthesis_across_sources",
        "whole_submission_holistic",
    }:
        return "analytical"
    if task_type in {
        "model_specification",
        "deterministic_derivation",
        "explanation_interpretation",
        "welfare_reasoning",
        "prediction_generation",
        "measurement_and_data_design",
        "regression_specification",
        "causal_identification",
    }:
        return "deterministic"
    text = f"{label}\n{marking_guidance}".lower()
    analytical_patterns = (
        "summary",
        "discuss",
        "evaluate",
        "commentary",
        "in your own words",
        "good idea",
        "critically assess",
        "justify your conclusion",
        "comment on whether",
        "compare and contrast",
    )
    deterministic_patterns = (
        "derive",
        "equation",
        "regression",
        "condition",
        "hypothesis",
        "measure",
        "data source",
        "calculate",
        "show carefully",
        "parameter",
        "endogeneity",
    )
    if any(pattern in text for pattern in deterministic_patterns):
        return "deterministic"
    if any(pattern in text for pattern in analytical_patterns):
        return "analytical"
    return "deterministic"


def infer_dependency_group(label: str, marking_guidance: str, parent_label: str = "") -> str:
    text = " ".join(marking_guidance.lower().split())
    direct_reference_patterns = (
        r"\busing\s+(?:your|the|previous|earlier)\s+(?:answer|result|results|work|analysis|model|derivation|prediction|predictions|evidence)\b",
        r"\busing\s+the\s+\w+\s+(?:model|results|answer|analysis)\s+from\s+(?:question|part|section)\s+\w+\b",
        r"\bfrom\s+(?:your|the|previous|earlier)\s+(?:answer|result|results|work|analysis|model|derivation|prediction|predictions|evidence)\b",
        r"\bbased on\s+(?:your|the|previous|earlier)\s+(?:answer|result|results|work|analysis|model|derivation|prediction|predictions|evidence)\b",
        r"\bgiven\s+(?:your|the)\s+(?:answer|result|results|work|analysis|model|derivation)\b",
        r"\bas\s+(?:shown|derived|established)\s+in\s+(?:your|the)\s+(?:answer|work|analysis)\b",
        r"\b(question|part|section)\s+\w+\s+(?:above|earlier|previously)\b",
        r"\byour\s+(?:answer|result|results|work|analysis)\s+to\s+(?:question|part|section)\s+\w+\b",
        r"\bprediction[s]?\s+from\s+your\s+answer\b",
        r"\bfrom\s+the\s+model\s+above\b",
    )
    if any(re.search(pattern, text) for pattern in direct_reference_patterns):
        return _normalize_label_key(parent_label or label)
    return ""


def build_unit_rubric_text(
    label: str,
    max_mark: float | None,
    grading_mode: str,
    task_type: str,
    question_text_exact: str,
    marking_guidance: str,
) -> str:
    unit = AssessmentUnit(
        label=label,
        max_mark=max_mark,
        grading_mode=grading_mode,
        task_type=task_type,
        question_text_exact=question_text_exact,
        marking_guidance=marking_guidance,
    )
    if grading_mode == "analytical":
        return build_analytical_rubric_text(unit)
    return build_deterministic_rubric_text(unit)


def build_deterministic_rubric_text(unit: AssessmentUnit) -> str:
    max_text = _format_number(unit.max_mark) if unit.max_mark is not None else "the available marks"
    criteria = _extract_classification_criteria(unit)
    return "\n".join(
        [
            f"Unit: {unit.label}",
            f"Task focus: {_build_task_focus(unit)}.",
            f"Available marks: 0 to {max_text}.",
            "Use this rubric to classify and score the attempt:",
            *[f"- {criterion}" for criterion in criteria],
            "- Award explicit partial credit for correct setup, valid intermediate steps, and correct interpretation even when the final answer is incomplete.",
            "- Place work lower when key steps are missing, algebra is wrong, the method is unjustified, or the interpretation does not follow from the working.",
            "- Rank attempts within the same band by accuracy, completeness of working, and clarity of interpretation.",
        ]
    ).strip()


def build_analytical_rubric_text(unit: AssessmentUnit) -> str:
    max_text = _format_number(unit.max_mark) if unit.max_mark is not None else "the available marks"
    criteria = _extract_classification_criteria(unit)
    return "\n".join(
        [
            f"Unit: {unit.label}",
            f"Task focus: {_build_task_focus(unit)}.",
            f"Available marks: 0 to {max_text}.",
            "Use UK-style quality bands: First, 2:1, 2:2, Third/Pass, Fail, Missing/unfinished.",
            "Use this rubric to classify and score the response:",
            *[f"- {criterion}" for criterion in criteria],
            "- Place work higher when the response is more precise, better supported, more relevant to the task, and more analytically convincing.",
            "- Place work lower when the response is vague, generic, descriptive, unsupported, or only partially relevant to the task.",
            "- Use the top band only for genuinely strong, well-supported work; do not compress all adequate answers into the middle of the scale.",
        ]
    ).strip()


def _render_unit_rubric_matrix(unit: AssessmentUnit) -> list[str]:
    max_text = _format_number(unit.max_mark) if unit.max_mark is not None else "variable"
    guidance_summary = _summarize_guidance_for_matrix(unit.marking_guidance)
    criteria_sentences = _extract_classification_criteria(unit)
    lines = [
        f"### {unit.label}",
        "",
        f"- Max mark: {max_text}",
        f"- Mode: {unit.grading_mode}",
        f"- Dependency group: {unit.dependency_group or 'independent'}",
        f"- Core criterion: {guidance_summary}",
        f"- Classification question: {_build_classification_question(unit)}",
        f"- Ranking rule: {_build_ranking_rule(unit)}",
        "",
        "#### Criteria Sentences",
        "",
    ]
    for sentence in criteria_sentences:
        lines.append(f"- {sentence}")
    lines.extend([
        "",
        "| Band | Descriptor | Within-band placement |",
        "| --- | --- | --- |",
    ])
    for band, descriptor, anchor in _band_rows_for_unit(unit):
        lines.append(f"| {band} | {descriptor} | {anchor} |")
    return lines


def _band_rows_for_unit(unit: AssessmentUnit) -> list[tuple[str, str, str]]:
    criterion = _build_task_focus(unit)
    if unit.grading_mode == "analytical":
        return [
            ("Missing/unfinished", f"Little or no usable response on {criterion}.", "Low: fragmentary or absent. Mid: partial attempt with minimal relevance. High: unfinished but shows some relevant substance."),
            ("Fail", f"Material is relevant in places but weak, unsupported, or substantially underdeveloped on {criterion}.", "Low: largely unsupported. Mid: some relevant points. High: close to pass but still too thin or inaccurate."),
            ("Third/Pass", f"Adequate response on {criterion} with basic relevance and limited development.", "Low: mostly descriptive. Mid: some explanation. High: coherent pass with clearer support."),
            ("2:2", f"Reasonably competent response on {criterion} with some support, but uneven precision or depth.", "Low: competent but patchy. Mid: solid lower-second standard. High: close to 2:1 but still uneven."),
            ("2:1", f"Strong response on {criterion} with clear support, relevant explanation, and useful appraisal.", "Low: strong core but some gaps. Mid: convincing upper-second standard. High: close to First with only limited weaknesses."),
            ("First", f"Excellent response on {criterion} that is precise, well-supported, and insightfully appraised.", "Low: clear First. Mid: strong First. High: rare exceptional answer with sustained precision and insight."),
        ]
    return [
        ("Missing", f"No usable attempt on {criterion}.", "Low: absent. Mid: fragmentary. High: minimal but relevant setup."),
        ("Weak", f"Response on {criterion} contains major errors or lacks the core method.", "Low: incorrect or missing method. Mid: some relevant setup. High: close to adequate but still materially wrong."),
        ("Adequate", f"Basic handling of {criterion} with some correct setup or partial method credit.", "Low: limited correct work. Mid: fair partial-credit answer. High: almost secure but with important gaps."),
        ("Secure", f"Mostly correct work on {criterion} with reasonable method and interpretation.", "Low: secure but patchy. Mid: reliable and mostly complete. High: close to strong with only minor issues."),
        ("Strong", f"Clear and largely correct handling of {criterion}, including accurate working and sensible interpretation where relevant.", "Low: strong but slightly uneven. Mid: consistently strong. High: nearly excellent."),
        ("Excellent", f"Full and accurate handling of {criterion} with precise method, complete reasoning, and convincing interpretation.", "Low: fully correct. Mid: fully correct and precise. High: exceptional clarity and completeness."),
    ]


def _summarize_guidance_for_matrix(marking_guidance: str) -> str:
    text = " ".join(marking_guidance.replace("\n", " ").split()).strip()
    if not text:
        return "Assess the unit against the stated task."
    match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    summary = match[0].strip()
    return summary[:220].rstrip()


def _extract_classification_criteria(unit: AssessmentUnit) -> list[str]:
    if unit.grading_mode == "analytical":
        return _build_analytical_criteria(unit)
    return _build_deterministic_criteria(unit)


def _default_classification_criteria(unit: AssessmentUnit) -> list[str]:
    if unit.grading_mode == "analytical":
        return [
            "Judge the quality of the argument, not just the amount written.",
            "Rank responses higher when they are more precise, better evidenced, and more analytically convincing.",
            "Use lower placement when the answer is descriptive, underdeveloped, unsupported, or only partially relevant.",
        ]
    return [
        "Judge the accuracy of the method, algebra, and final result.",
        "Rank responses higher when they show more complete working, fewer errors, and stronger interpretation.",
        "Use lower placement when important steps are missing or the reasoning is materially incorrect.",
    ]


def _build_analytical_criteria(unit: AssessmentUnit) -> list[str]:
    task_type_criteria = _criteria_for_task_type(unit)
    if task_type_criteria:
        return task_type_criteria[:5]
    text = unit.marking_guidance.lower()
    focus = _build_task_focus(unit)
    criteria = [
        f"Evaluate whether the response directly addresses {focus}.",
        "Evaluate whether the claims are accurate, relevant, and clearly explained rather than generic or impressionistic.",
        "Evaluate whether the answer uses supporting evidence, references, or course concepts where the task requires them.",
        "Evaluate whether the response shows appraisal, qualification, or correction rather than simple description.",
    ]
    if any(term in text for term in ("quote", "page", "slide", "reference", "evidence", "report")):
        criteria[2] = "Evaluate whether the answer uses specific supporting evidence, references, page numbers, or cited material where required."
    if any(term in text for term in ("rewrite", "own tone", "own words", "edit the output", "correct any mistakes")):
        criteria.append("Evaluate whether the response corrects mistakes and rewrites the material clearly in the student's own justified form.")
    return criteria[:5]


def _build_deterministic_criteria(unit: AssessmentUnit) -> list[str]:
    task_type_criteria = _criteria_for_task_type(unit)
    if task_type_criteria:
        return task_type_criteria[:5]
    text = unit.marking_guidance.lower()
    focus = _build_task_focus(unit)
    criteria = [
        f"Evaluate whether the attempt sets up and answers {focus} correctly.",
        "Evaluate whether the working is complete enough to justify the result, not just whether a final answer is stated.",
        "Evaluate whether the algebra, notation, equation choice, or derivation steps are accurate.",
        "Evaluate whether any explanation or interpretation follows correctly from the method and result.",
    ]
    if any(term in text for term in ("measure", "variable", "data source", "regression", "equation")):
        criteria.append("Evaluate whether variables, measurement choices, or formal specifications are clearly and correctly defined.")
    return criteria[:5]


def _build_classification_question(unit: AssessmentUnit) -> str:
    if unit.grading_mode == "analytical":
        return f"Which quality band best fits the student's response to {unit.label}, and where does it sit inside that band?"
    return f"Which correctness band best fits the student's attempt on {unit.label}, and where does it sit inside that band?"


def _build_ranking_rule(unit: AssessmentUnit) -> str:
    if unit.grading_mode == "analytical":
        return "Group scripts by quality band first, then rank them within the band by precision, support, relevance, and analytical depth."
    return "Group attempts by correctness band first, then rank them within the band by method accuracy, completeness of working, and interpretation."


def _criteria_for_task_type(unit: AssessmentUnit) -> list[str]:
    focus = _build_task_focus(unit)
    task_type = unit.task_type
    if task_type == "critique_and_revision":
        return [
            f"Evaluate whether the response completes {focus}.",
            "Evaluate whether the answer identifies substantive mistakes, omissions, or distortions rather than offering generic criticism.",
            "Evaluate whether the critique is grounded in the supplied source, prompt, or task requirements.",
            "Evaluate whether the revised version is clearer, more accurate, and appropriately rewritten rather than lightly edited.",
        ]
    if task_type == "model_specification":
        return [
            f"Evaluate whether the response states {focus} correctly.",
            "Evaluate whether symbols, terms, and relationships are defined accurately and consistently.",
            "Evaluate whether the explanation makes clear what each part of the specification means in context.",
            "Evaluate whether any interpretation follows from the stated model rather than from unsupported commentary.",
        ]
    if task_type == "deterministic_derivation":
        return [
            f"Evaluate whether the attempt sets up and answers {focus} correctly.",
            "Evaluate whether the working is complete enough to justify the result, not just whether a final answer is stated.",
            "Evaluate whether derivation steps, algebra, notation, and conditions are accurate.",
            "Evaluate whether any interpretation follows correctly from the method and result.",
        ]
    if task_type == "explanation_interpretation":
        return [
            f"Evaluate whether the response explains {focus} clearly and accurately.",
            "Evaluate whether the explanation identifies the relevant mechanism, direction, or relationship instead of describing outcomes vaguely.",
            "Evaluate whether the interpretation matches the underlying model, evidence, or setup.",
            "Evaluate whether the reasoning is complete enough to show why the conclusion follows.",
        ]
    if task_type == "welfare_reasoning":
        return [
            f"Evaluate whether the response addresses {focus} using the relevant conditions or assumptions.",
            "Evaluate whether the answer distinguishes effects on the relevant parties, components, or margins rather than collapsing them into a single claim.",
            "Evaluate whether the welfare reasoning is internally consistent and follows from the setup.",
            "Evaluate whether the conclusion is qualified appropriately when conditions or tradeoffs matter.",
        ]
    if task_type == "evaluative_discussion":
        return [
            f"Evaluate whether the response directly addresses {focus}.",
            "Evaluate whether the answer makes a defensible judgement rather than offering only description or summary.",
            "Evaluate whether the reasoning is supported with relevant evidence, examples, or concepts where the task requires them.",
            "Evaluate whether the discussion acknowledges tradeoffs, limits, counterarguments, or qualifications where they matter.",
        ]
    if task_type == "prediction_generation":
        return [
            f"Evaluate whether the response states {focus} that follow from the setup.",
            "Evaluate whether each prediction is directionally clear and not merely a repetition of the question.",
            "Evaluate whether the predictions are tied to the stated model, assumptions, or mechanism.",
            "Evaluate whether the answer distinguishes strong predictions from conditional or qualified ones where necessary.",
        ]
    if task_type == "measurement_and_data_design":
        return [
            f"Evaluate whether the response defines {focus} clearly.",
            "Evaluate whether the answer maps concepts to observable variables or indicators in a plausible way.",
            "Evaluate whether proposed data sources are suitable for the measures being discussed.",
            "Evaluate whether the response explains how the source or measure captures the intended concept and notes material limitations where relevant.",
        ]
    if task_type == "regression_specification":
        return [
            f"Evaluate whether the response states {focus} correctly.",
            "Evaluate whether variables, terms, and functional relationships are defined clearly and consistently.",
            "Evaluate whether the coefficient, comparison, or hypothesis of interest is identified correctly.",
            "Evaluate whether the specification matches the stated empirical question rather than a different one.",
        ]
    if task_type == "causal_identification":
        return [
            f"Evaluate whether the response identifies {focus} correctly.",
            "Evaluate whether the answer explains the mechanism behind the identification problem rather than naming it only.",
            "Evaluate whether proposed remedies are relevant, feasible, and connected to the stated problem.",
            "Evaluate whether the response distinguishes stronger and weaker solutions or remaining limitations where relevant.",
        ]
    if task_type == "synthesis_across_sources":
        return [
            f"Evaluate whether the response completes {focus}.",
            "Evaluate whether the answer identifies the main claims, assumptions, or omissions across the relevant sources or materials.",
            "Evaluate whether theory, evidence, and source content are integrated rather than discussed in isolation.",
            "Evaluate whether the response explains what the model or framework can and cannot account for.",
        ]
    return []


def _build_task_focus(unit: AssessmentUnit) -> str:
    if unit.task_type == "critique_and_revision":
        return "the required AI-generated summary, source-based accuracy audit, and corrected rewrite"
    if unit.task_type == "model_specification":
        return "the required payoff function and explanation of its terms"
    if unit.task_type == "deterministic_derivation":
        return "the required derivation and resulting conditions"
    if unit.task_type == "explanation_interpretation":
        return "the required explanation and interpretation"
    if unit.task_type == "welfare_reasoning":
        return "the relevant welfare conditions and effects on the affected parties"
    if unit.task_type == "evaluative_discussion":
        return "the required evaluative discussion"
    if unit.task_type == "prediction_generation":
        return "the model's predictions"
    if unit.task_type == "measurement_and_data_design":
        return "the required measurement choices and data-source mapping"
    if unit.task_type == "regression_specification":
        return "the required regression equation, coefficient of interest, and hypothesis"
    if unit.task_type == "causal_identification":
        return "the endogeneity concerns and feasible solutions"
    if unit.task_type == "synthesis_across_sources":
        return "the required synthesis of theory, evidence, and source claims"
    text = unit.marking_guidance.lower()
    if unit.grading_mode == "analytical":
        if any(term in text for term in ("comment", "evaluate", "discuss", "good idea")):
            return "the required evaluative discussion"
        if any(term in text for term in ("rewrite", "own tone", "own words")):
            return "the required corrected and rewritten response"
        return "the required analytical discussion"
    if any(term in text for term in ("equation", "regression")):
        return "the required equation or formal specification"
    if any(term in text for term in ("derive", "condition")):
        return "the required derivation and resulting conditions"
    if any(term in text for term in ("measure", "variable", "data source")):
        return "the required measurement or variable definition"
    if any(term in text for term in ("prediction", "list")):
        return "the requested predictions or listed implications"
    if any(term in text for term in ("explain", "parameter", "intuition", "interpret")):
        return "the required explanation and interpretation"
    return "the required analytical step"


def normalize_part_analysis(data: dict[str, Any], part: SubmissionPart) -> dict[str, Any]:
    expected_label = part.label
    score = data.get("provisional_score")
    score_field = "provisional_score"
    if score is None:
        score = data.get("provisional_score_0_to_100")
        score_field = "provisional_score_0_to_100"
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"Model did not return a numeric provisional score for {expected_label}: {data}")
    normalized_score = float(score)
    if normalized_score < 0:
        raise ValueError(f"Model returned a negative provisional score for {expected_label}: {data}")
    if score_field == "provisional_score_0_to_100" and normalized_score > 100:
        raise ValueError(f"Model returned a provisional score outside 0-100 for {expected_label}: {data}")

    strengths = _normalize_string_list(data.get("strengths"), "strengths", expected_label, minimum=2)
    weaknesses = _normalize_string_list(data.get("weaknesses"), "weaknesses", expected_label, minimum=2)
    evidence = _normalize_string_list(data.get("evidence"), "evidence", expected_label, minimum=1)
    coverage_comment = data.get("coverage_comment")
    if not isinstance(coverage_comment, str) or not coverage_comment.strip():
        raise ValueError(f"Model did not return a coverage_comment for {expected_label}: {data}")

    if score_field == "provisional_score" and part.max_mark is not None and part.max_mark > 0:
        score_pct = (normalized_score / float(part.max_mark)) * 100.0
    else:
        score_pct = normalized_score
    grade_band = _normalize_grade_band(
        value=data.get("grade_band"),
        score_pct=score_pct,
        marking_guidance=part.marking_guidance,
    )
    within_band_position = _normalize_within_band_position(
        value=data.get("within_band_position"),
        score_pct=score_pct,
        grade_band=grade_band,
        marking_guidance=part.marking_guidance,
    )
    band_confidence = data.get("band_confidence_0_to_100")
    if isinstance(band_confidence, bool) or not isinstance(band_confidence, (int, float)):
        band_confidence_value = _default_band_confidence(grade_band, marking_guidance=part.marking_guidance)
    else:
        band_confidence_value = round(float(band_confidence), 2)
        if band_confidence_value < 0 or band_confidence_value > 100:
            band_confidence_value = _default_band_confidence(grade_band, marking_guidance=part.marking_guidance)

    return {
        "section_label": expected_label,
        "provisional_score_0_to_100": round(normalized_score, 2) if score_field == "provisional_score_0_to_100" else None,
        "provisional_score": round(normalized_score, 2) if score_field == "provisional_score" else None,
        "grade_band": grade_band,
        "within_band_position": within_band_position,
        "band_confidence_0_to_100": band_confidence_value,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "evidence": evidence,
        "coverage_comment": coverage_comment.strip(),
    }


def _run_part_analysis_with_retry(
    part: SubmissionPart,
    context: MarkingContext,
    filename: str,
    model_name: str,
    ollama_url: str,
) -> dict[str, Any]:
    messages = build_part_messages(part, context, filename)
    last_error: Exception | None = None
    for attempt in range(2):
        analysis = _call_ollama_json(
            model_name=model_name,
            messages=messages,
            ollama_url=ollama_url,
        )
        try:
            return normalize_part_analysis(analysis, part=part)
        except ValueError as exc:
            last_error = exc
            if attempt == 1 or not _should_retry_part_analysis(exc):
                raise
            messages = _build_part_analysis_retry_messages(messages, part, exc)
    if last_error is not None:
        raise last_error
    raise ValueError(f"Part analysis failed without a captured error for {part.label}.")


def build_missing_part_analysis(part: SubmissionPart) -> dict[str, Any]:
    guidance = part.marking_guidance
    zero_score = 0.0
    grade_band = _normalize_grade_band(
        value="missing/unfinished",
        score_pct=zero_score,
        marking_guidance=guidance,
    )
    within_band_position = _normalize_within_band_position(
        value="low",
        score_pct=zero_score,
        grade_band=grade_band,
        marking_guidance=guidance,
    )
    max_mark_text = _format_number(part.max_mark) if part.max_mark is not None else "100"
    return {
        "section_label": part.label,
        "provisional_score_0_to_100": None,
        "provisional_score": zero_score,
        "grade_band": grade_band,
        "within_band_position": within_band_position,
        "band_confidence_0_to_100": 95.0,
        "strengths": [
            "No creditable response was identified for this section.",
            "The section remains available for explicit zero-credit accounting in the final arithmetic.",
        ],
        "weaknesses": [
            "The expected section content was not found in the extracted submission text.",
            f"No valid method, evidence, or answer was available to earn marks out of {max_mark_text}.",
        ],
        "evidence": [
            "No section text was matched to this expected assessment unit.",
        ],
        "coverage_comment": (
            "This expected assessment unit was not detected in the submission text. "
            "It is therefore treated as missing for deterministic arithmetic and awarded zero."
        ),
    }


def moderate_part_analyses_across_submission(
    script_text: str,
    context: MarkingContext,
    filename: str,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[dict[str, Any]]:
    data = _call_ollama_json(
        model_name=model_name,
        messages=build_moderation_messages(script_text, parts, part_analyses, context, filename),
        ollama_url=ollama_url,
    )
    return normalize_moderated_part_scores(data, parts, part_analyses)


def moderate_linked_part_analyses(
    script_text: str,
    context: MarkingContext,
    filename: str,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    assessment_map: AssessmentMap,
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[dict[str, Any]]:
    unit_by_key = {_normalize_label_key(unit.label): unit for unit in assessment_map.units}
    groups: dict[str, list[int]] = {}
    for index, part in enumerate(parts):
        unit = unit_by_key.get(_normalize_label_key(part.label))
        if unit is None or not unit.dependency_group:
            continue
        groups.setdefault(unit.dependency_group, []).append(index)

    if not groups:
        return part_analyses

    updated = [dict(item) for item in part_analyses]
    for _, indexes in groups.items():
        if len(indexes) < 2:
            continue
        grouped_parts = [parts[index] for index in indexes]
        grouped_analyses = [updated[index] for index in indexes]
        should_moderate, _ = _moderation_group_decision(grouped_parts, grouped_analyses)
        if not should_moderate:
            continue
        moderated = moderate_part_analyses_across_submission(
            script_text=script_text,
            context=context,
            filename=filename,
            parts=grouped_parts,
            part_analyses=grouped_analyses,
            model_name=model_name,
            ollama_url=ollama_url,
        )
        for index, moderated_item in zip(indexes, moderated):
            updated[index] = moderated_item
    return updated


def _should_moderate_group(
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
) -> bool:
    return _moderation_group_decision(parts, part_analyses)[0]


def describe_moderation_plan(
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    assessment_map: AssessmentMap,
) -> list[dict[str, Any]]:
    unit_by_key = {_normalize_label_key(unit.label): unit for unit in assessment_map.units}
    groups: dict[str, list[int]] = {}
    for index, part in enumerate(parts):
        unit = unit_by_key.get(_normalize_label_key(part.label))
        if unit is None or not unit.dependency_group:
            continue
        groups.setdefault(unit.dependency_group, []).append(index)
    plan: list[dict[str, Any]] = []
    for group_name, indexes in groups.items():
        grouped_parts = [parts[index] for index in indexes]
        grouped_analyses = [part_analyses[index] for index in indexes]
        should_moderate, reason = _moderation_group_decision(grouped_parts, grouped_analyses)
        plan.append(
            {
                "dependency_group": group_name,
                "part_labels": [part.label for part in grouped_parts],
                "should_moderate": should_moderate,
                "reason": reason,
            }
        )
    return plan


def _moderation_group_decision(
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
) -> tuple[bool, str]:
    if len(parts) < 2 or len(parts) != len(part_analyses):
        return False, "insufficient_group_size"
    normalized_scores: list[float] = []
    band_levels: list[int] = []
    for part, analysis in zip(parts, part_analyses):
        if not part.section_text.strip():
            return False, f"missing_section_text:{part.label}"
        score_pct = _analysis_score_pct(part, analysis)
        if score_pct is None:
            return False, f"missing_score:{part.label}"
        normalized_scores.append(score_pct)
        band_level = _band_level(str(analysis.get("grade_band", "")).strip())
        if band_level is not None:
            band_levels.append(band_level)
    spread = max(normalized_scores) - min(normalized_scores)
    if spread > 15.0:
        return False, f"wide_score_spread:{spread:.2f}"
    if band_levels and (max(band_levels) - min(band_levels)) > 1:
        return False, "wide_band_gap"
    return True, "eligible"


def _analysis_score_pct(part: SubmissionPart, analysis: dict[str, Any]) -> float | None:
    score = analysis.get("provisional_score")
    if score is not None:
        if part.max_mark is None or float(part.max_mark) <= 0:
            return None
        return round((float(score) / float(part.max_mark)) * 100.0, 2)
    score_pct = analysis.get("provisional_score_0_to_100")
    if score_pct is None:
        return None
    return round(float(score_pct), 2)


def _band_level(grade_band: str) -> int | None:
    if not grade_band:
        return None
    order = {
        "Missing/unfinished": 0,
        "Missing": 0,
        "Fail": 1,
        "Weak": 1,
        "Third/Pass": 2,
        "Developing": 2,
        "2:2": 3,
        "Adequate": 3,
        "2:1": 4,
        "Secure": 4,
        "First": 5,
        "Strong": 5,
        "Excellent": 6,
    }
    return order.get(grade_band)


def normalize_moderated_part_scores(
    data: dict[str, Any],
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    adjusted_sections = data.get("adjusted_sections")
    if not isinstance(adjusted_sections, list):
        raise ValueError(f"Model returned invalid adjusted_sections for moderation: {data}")

    current_by_key = {_normalize_label_key(analysis["section_label"]): dict(analysis) for analysis in part_analyses}
    part_by_key = {_normalize_label_key(part.label): part for part in parts}
    original_by_key = {_normalize_label_key(analysis["section_label"]): analysis for analysis in part_analyses}
    moderated: dict[str, dict[str, Any]] = {}

    for item in adjusted_sections:
        if not isinstance(item, dict):
            raise ValueError(f"Moderation item was not an object: {item}")
        label = str(item.get("section_label", "")).strip()
        key = _normalize_label_key(label)
        if key not in current_by_key or key not in part_by_key:
            raise ValueError(f"Moderation returned an unknown section label: {label}")
        current = dict(current_by_key[key])
        part = part_by_key[key]
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Moderation did not return a rationale for {label}: {item}")

        adjusted_abs = item.get("adjusted_provisional_score")
        adjusted_pct = item.get("adjusted_provisional_score_0_to_100")
        if adjusted_abs is not None and adjusted_pct is not None:
            raise ValueError(f"Moderation returned both absolute and percentage scores for {label}: {item}")

        original_score = original_by_key[key]
        before = original_score.get("provisional_score")
        if before is None:
            before = original_score.get("provisional_score_0_to_100")

        if part.max_mark is not None:
            if adjusted_abs is None:
                adjusted_value = float(before)
            else:
                if isinstance(adjusted_abs, bool) or not isinstance(adjusted_abs, (int, float)):
                    raise ValueError(f"Moderation did not return an absolute score for {label}: {item}")
                adjusted_value = round(float(adjusted_abs), 2)
                if adjusted_value < 0 or adjusted_value > float(part.max_mark):
                    raise ValueError(f"Moderation returned out-of-range score for {label}: {item}")
            current["provisional_score"] = adjusted_value
            current["provisional_score_0_to_100"] = None
        else:
            if adjusted_pct is None:
                adjusted_value = float(before)
            else:
                if isinstance(adjusted_pct, bool) or not isinstance(adjusted_pct, (int, float)):
                    raise ValueError(f"Moderation did not return a percentage score for {label}: {item}")
                adjusted_value = round(float(adjusted_pct), 2)
                if adjusted_value < 0 or adjusted_value > 100:
                    raise ValueError(f"Moderation returned out-of-range percentage for {label}: {item}")
            current["provisional_score"] = None
            current["provisional_score_0_to_100"] = adjusted_value

        after = current.get("provisional_score")
        if after is None:
            after = current.get("provisional_score_0_to_100")
        current["moderation_rationale"] = rationale.strip()
        current["moderation_delta"] = round(float(after) - float(before), 2)
        moderated[key] = current

    for analysis in part_analyses:
        key = _normalize_label_key(analysis["section_label"])
        if key in moderated:
            continue
        preserved = dict(analysis)
        preserved["moderation_rationale"] = "No moderation change returned."
        preserved["moderation_delta"] = 0.0
        moderated[key] = preserved

    return [moderated[_normalize_label_key(part.label)] for part in parts]


def normalize_marking_result(
    data: dict[str, Any],
    expected_max_mark: float,
    diagnostics: SubmissionDiagnostics | None = None,
    parts: list[SubmissionPart] | None = None,
    part_analyses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    feedback = _normalize_feedback_text(data.get("overall_feedback") or data.get("feedback"))
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError(f"Model did not return feedback text: {data}")

    total_mark = data.get("total_mark")
    model_max_mark = data.get("max_mark")
    if isinstance(model_max_mark, bool):
        raise ValueError(f"Model returned an invalid max_mark: {data}")

    normalized_total = None
    normalized_max = expected_max_mark

    if total_mark is not None:
        if isinstance(total_mark, bool) or not isinstance(total_mark, (int, float)):
            raise ValueError(f"Model did not return a numeric total_mark: {data}")
        normalized_total = float(total_mark)

    if isinstance(model_max_mark, (int, float)):
        if normalized_total is None:
            raise ValueError(f"Model returned max_mark without a numeric total_mark: {data}")
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
    elif normalized_total is not None and normalized_total > expected_max_mark and normalized_total <= 100 and expected_max_mark != 100:
        normalized_total = round((normalized_total * expected_max_mark) / 100.0, 2)

    if normalized_total is not None and (normalized_total < 0 or normalized_total > expected_max_mark):
        raise ValueError(
            "Model returned a total_mark outside the expected range: "
            f"{_format_number(normalized_total)} / {_format_number(expected_max_mark)}."
        )

    strengths_value = data.get("strengths")
    weaknesses_value = data.get("weaknesses")
    strengths = _normalize_string_list(strengths_value, "strengths", "final result", minimum=2) if strengths_value is not None else []
    weaknesses = _normalize_string_list(weaknesses_value, "weaknesses", "final result", minimum=2) if weaknesses_value is not None else []

    covered_parts_raw = data.get("covered_parts")
    covered_parts = _normalize_string_list(covered_parts_raw, "covered_parts", "final result", minimum=1) if covered_parts_raw is not None else []
    if parts:
        covered_parts = [part.label for part in parts]

    if len(feedback.split()) < 120:
        raise ValueError("Model returned feedback that is shorter than the required minimum length.")

    if normalized_total is not None and normalized_total == expected_max_mark and _has_major_weaknesses(weaknesses, feedback):
        raise ValueError("Model awarded full marks while also describing substantial weaknesses.")

    if diagnostics and diagnostics.low_text:
        strengths.append(f"Extraction warning: submission contains only {diagnostics.extracted_word_count} words.")

    validation_notes = []
    if diagnostics:
        if diagnostics.possible_extraction_issue:
            validation_notes.append("possible_extraction_issue")
        if diagnostics.low_text:
            validation_notes.append("low_text")

    ai_total_mark = _clean_number(normalized_total) if normalized_total is not None else None
    math_total_mark = None
    ai_math_mark_delta = None

    if part_analyses:
        provisional_scores = [
            float(item["provisional_score"] if item.get("provisional_score") is not None else item["provisional_score_0_to_100"])
            for item in part_analyses
        ]
        moderation_deltas = [abs(float(item.get("moderation_delta", 0.0))) for item in part_analyses if item.get("moderation_delta") is not None]
        spread = max(provisional_scores) - min(provisional_scores)
        validation_notes.append(f"part_score_spread={spread:.2f}")
        if moderation_deltas:
            validation_notes.append(f"part_moderation_max_delta={max(moderation_deltas):.2f}")
        math_total = compute_total_mark_from_part_scores(expected_max_mark, parts or [], part_analyses)
        if math_total is None:
            raise ValueError("Could not compute deterministic total from part scores.")
        else:
            math_total_mark = _clean_number(math_total)
            normalized_total = math_total
            validation_notes.append("math_total_used")
            if ai_total_mark is not None:
                ai_math_mark_delta = _clean_number(round(ai_total_mark - math_total_mark, 2))
            if ai_total_mark is not None and abs(ai_total_mark - math_total_mark) > 1e-9:
                validation_notes.append(f"ai_total_disagreement={abs(ai_total_mark - math_total_mark):.2f}")
    elif normalized_total is None:
        raise ValueError("No deterministic part scores were available and the model did not return a total_mark.")

    return {
        "total_mark": _clean_number(normalized_total),
        "max_mark": _clean_number(normalized_max),
        "ai_total_mark": ai_total_mark,
        "math_total_mark": math_total_mark,
        "ai_math_mark_delta": ai_math_mark_delta,
        "overall_feedback": feedback.strip(),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "covered_parts": covered_parts,
        "detected_part_count": diagnostics.detected_part_count if diagnostics else 1,
        "detected_part_labels": list(diagnostics.detected_part_labels) if diagnostics else [],
        "extracted_word_count": diagnostics.extracted_word_count if diagnostics else len(feedback.split()),
        "possible_extraction_issue": diagnostics.possible_extraction_issue if diagnostics else False,
        "validation_notes": validation_notes,
    }


def build_local_final_result(
    context: MarkingContext,
    diagnostics: SubmissionDiagnostics,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    feedback = _render_feedback_from_part_analyses(parts, part_analyses, context.max_mark)
    strengths = _aggregate_part_bullets(part_analyses, field="strengths")
    weaknesses = _aggregate_part_bullets(part_analyses, field="weaknesses")
    data = {
        "overall_feedback": feedback,
        "covered_parts": [part.label for part in parts],
        "strengths": strengths[: max(2, len(strengths))],
        "weaknesses": weaknesses[: max(2, len(weaknesses))],
    }
    return normalize_marking_result(
        data,
        expected_max_mark=context.max_mark,
        diagnostics=diagnostics,
        parts=parts,
        part_analyses=part_analyses,
    )


def _attach_latency_metrics(
    result: dict[str, Any],
    timings: dict[str, float],
    total_started: float,
    parts: list[SubmissionPart],
) -> dict[str, Any]:
    total_seconds = round(time.perf_counter() - total_started, 2)
    part_count = max(len(parts), 1)
    result["latency_seconds_total"] = total_seconds
    result["latency_seconds_assessment_prepare"] = round(timings.get("assessment_prepare", 0.0), 2)
    result["latency_seconds_rubric_verify"] = result["latency_seconds_assessment_prepare"]
    result["latency_seconds_structure_detect"] = round(timings.get("structure_detect", 0.0), 2)
    result["latency_seconds_part_refine"] = round(timings.get("part_refine", 0.0), 2)
    result["latency_seconds_part_analysis"] = round(timings.get("part_analysis", 0.0), 2)
    result["latency_seconds_moderation"] = round(timings.get("moderation", 0.0), 2)
    result["latency_seconds_finalize"] = round(timings.get("finalize", 0.0), 2)
    result["latency_seconds_part_analysis_per_part_avg"] = round(result["latency_seconds_part_analysis"] / part_count, 2)
    validation_notes = list(result.get("validation_notes", []))
    validation_notes.append(f"latency_script_seconds={total_seconds:.2f}")
    validation_notes.append(f"latency_assessment_prepare_seconds={result['latency_seconds_assessment_prepare']:.2f}")
    validation_notes.append(f"latency_part_analysis_seconds={result['latency_seconds_part_analysis']:.2f}")
    result["validation_notes"] = validation_notes
    return result


def compute_total_mark_from_part_scores(
    expected_max_mark: float,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
) -> float | None:
    if not parts or not part_analyses or len(parts) != len(part_analyses):
        return None

    part_marks = []
    for part, analysis in zip(parts, part_analyses):
        effective_part_max = float(part.max_mark) if part.max_mark is not None else None
        if len(parts) == 1 and effective_part_max is None:
            effective_part_max = float(expected_max_mark)

        score = analysis.get("provisional_score")
        if score is not None:
            part_marks.append(float(score))
            continue

        score_100 = analysis.get("provisional_score_0_to_100")
        if score_100 is None:
            return None
        if effective_part_max is None:
            return None
        part_marks.append((float(score_100) * effective_part_max) / 100.0)

    if not part_marks:
        return None

    summed_total = sum(part_marks)
    summed_total = round(summed_total, 2)
    return min(expected_max_mark, max(0.0, summed_total))


def _aggregate_part_bullets(part_analyses: list[dict[str, Any]], field: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for analysis in part_analyses:
        for raw in analysis.get(field, []):
            text = str(raw).strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            items.append(text)
            if len(items) >= 6:
                return items
    return items


def _render_feedback_from_part_analyses(
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    expected_max_mark: float,
) -> str:
    total = compute_total_mark_from_part_scores(expected_max_mark, parts, part_analyses)
    total_text = _format_number(total if total is not None else expected_max_mark)
    intro = (
        f"This script is classified from part-level evidence rather than a separate final-language pass. "
        f"The deterministic overall mark is {total_text} out of {_format_number(expected_max_mark)}, computed as the sum of the part scores."
    )
    section_lines: list[str] = []
    for part, analysis in zip(parts, part_analyses):
        section_score = analysis.get("provisional_score")
        if section_score is None and analysis.get("provisional_score_0_to_100") is not None and part.max_mark is not None:
            section_score = round((float(analysis["provisional_score_0_to_100"]) * float(part.max_mark)) / 100.0, 2)
        score_text = _format_number(float(section_score)) if section_score is not None else "unscored"
        max_text = _format_number(float(part.max_mark)) if part.max_mark is not None else "100"
        band = str(analysis.get("grade_band", "classified")).strip()
        position = str(analysis.get("within_band_position", "mid")).strip()
        coverage = str(analysis.get("coverage_comment", "")).strip()
        top_strength = ", ".join(str(item).strip() for item in analysis.get("strengths", [])[:2] if str(item).strip())
        top_weakness = ", ".join(str(item).strip() for item in analysis.get("weaknesses", [])[:2] if str(item).strip())
        section_lines.append(
            f"{part.label} scored {score_text}/{max_text} and was placed in the {band} band at the {position} end of that bracket. "
            f"{coverage} Stronger features here include {top_strength or 'clear relevant material'}, while the mark is held back by {top_weakness or 'remaining weaknesses in precision and support'}."
        )
    outro = (
        "The feedback therefore follows the classified rubric language for each part, rather than trying to invent a fresh overall judgement. "
        "Where a section sits inside its bracket depends on the evidence quality, completeness, precision, and support actually visible in that section."
    )
    return " ".join([intro, *section_lines, outro]).strip()


def calibrate_marks_across_students(
    provisional_results: list[dict[str, Any]],
    model_name: str,
    model_label: str,
    context: MarkingContext,
    assessment_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, dict[str, Any]]:
    eligible = []
    for item in provisional_results:
        mark = item.get("total_mark")
        if isinstance(mark, (int, float)):
            eligible.append(
                {
                    "filename": str(item.get("filename", "")),
                    "provisional_mark": float(mark),
                    "detected_part_count": item.get("detected_part_count"),
                    "strengths": list(item.get("strengths", []))[:2],
                    "weaknesses": list(item.get("weaknesses", []))[:2],
                    "validation_notes": list(item.get("validation_notes", [])),
                }
            )

    if len(eligible) < 2:
        return {}

    data = _call_ollama_json(
        model_name=model_name,
        messages=build_calibration_messages(assessment_name, context, model_label, eligible),
        ollama_url=ollama_url,
    )
    return normalize_calibration_result(data, eligible, context.max_mark)


def normalize_calibration_result(
    data: dict[str, Any],
    provisional_results: list[dict[str, Any]],
    max_mark: float,
) -> dict[str, dict[str, Any]]:
    calibrated = data.get("calibrated_results")
    if not isinstance(calibrated, list):
        raise ValueError(f"Calibration response did not include calibrated_results: {data}")

    expected = {str(item["filename"]): float(item["provisional_mark"]) for item in provisional_results}
    resolved: dict[str, dict[str, Any]] = {}
    for item in calibrated:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "")).strip()
        adjusted_mark = item.get("adjusted_mark")
        rationale = str(item.get("rationale", "")).strip()
        if not filename or filename not in expected:
            continue
        if isinstance(adjusted_mark, bool) or not isinstance(adjusted_mark, (int, float)):
            raise ValueError(f"Calibration returned a non-numeric adjusted mark for {filename}: {item}")
        adjusted = float(adjusted_mark)
        if adjusted < 0 or adjusted > max_mark:
            raise ValueError(f"Calibration returned an out-of-range mark for {filename}: {item}")
        resolved[filename] = {
            "adjusted_mark": _clean_number(adjusted),
            "delta": round(adjusted - expected[filename], 2),
            "rationale": rationale,
        }

    missing = [filename for filename in expected if filename not in resolved]
    if missing:
        raise ValueError("Calibration did not return marks for: " + ", ".join(missing))
    return resolved


def verify_marking_result(
    script_text: str,
    context: MarkingContext,
    filename: str,
    grading_result: dict[str, Any],
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    data = _call_ollama_json(
        model_name=model_name,
        messages=build_verification_messages(script_text, context, filename, grading_result),
        ollama_url=ollama_url,
    )
    return normalize_verification_result(data)


def regrade_marking_result(
    script_text: str,
    context: MarkingContext,
    filename: str,
    prior_result: dict[str, Any],
    verifier_result: dict[str, Any],
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> dict[str, Any]:
    parts = [
        SubmissionPart(label=label, focus_hint="")
        for label in prior_result.get("covered_parts", []) or ["Whole Submission"]
    ]
    diagnostics = build_submission_diagnostics(script_text, parts)
    messages = build_regrade_messages(script_text, context, filename, prior_result, verifier_result)
    return _run_final_result_with_retry(
        model_name=model_name,
        messages=messages,
        expected_max_mark=context.max_mark,
        diagnostics=diagnostics,
        parts=parts,
        part_analyses=[],
        ollama_url=ollama_url,
    )


def normalize_verification_result(data: dict[str, Any]) -> dict[str, Any]:
    agreement = str(data.get("agreement", "")).strip()
    if agreement not in {"agree", "minor_concern", "major_concern"}:
        raise ValueError(f"Verifier returned an invalid agreement value: {data}")
    confidence = data.get("confidence_0_to_100")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"Verifier returned an invalid confidence score: {data}")
    confidence_value = float(confidence)
    if confidence_value < 0 or confidence_value > 100:
        raise ValueError(f"Verifier confidence is out of range: {data}")
    issues = data.get("issues")
    if not isinstance(issues, list):
        raise ValueError(f"Verifier returned invalid issues: {data}")
    normalized_issues = [str(item).strip() for item in issues if str(item).strip()]
    recommendation = str(data.get("recommendation", "")).strip()
    if not recommendation:
        raise ValueError(f"Verifier returned an empty recommendation: {data}")
    return {
        "agreement": agreement,
        "confidence_0_to_100": round(confidence_value, 2),
        "issues": normalized_issues,
        "recommendation": recommendation,
    }


def list_submission_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES)


def parse_keyword_patterns(value: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def discover_assessment_bundles(
    root_folder: Path,
    rubric_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["rubric"],
    brief_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["brief"],
    marking_scheme_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["marking_scheme"],
    graded_sample_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["graded_sample"],
    other_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["other"],
) -> list[AssessmentBundle]:
    if not root_folder.exists():
        raise ValueError(f"Assessment root folder not found: {root_folder}")
    if not root_folder.is_dir():
        raise ValueError(f"Assessment root path is not a folder: {root_folder}")

    bundles: list[AssessmentBundle] = []
    for folder in sorted(path for path in root_folder.iterdir() if path.is_dir()):
        bundles.append(
            build_assessment_bundle(
                folder=folder,
                rubric_keywords=rubric_keywords,
                brief_keywords=brief_keywords,
                marking_scheme_keywords=marking_scheme_keywords,
                graded_sample_keywords=graded_sample_keywords,
                other_keywords=other_keywords,
            )
        )
    return bundles


def build_single_assessment_bundle(
    folder: Path,
    assessment_name: str | None = None,
    rubric_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["rubric"],
    brief_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["brief"],
    marking_scheme_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["marking_scheme"],
    graded_sample_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["graded_sample"],
    other_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["other"],
) -> AssessmentBundle:
    if not folder.exists():
        raise ValueError(f"Assessment folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Assessment path is not a folder: {folder}")

    top_level_files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES)
    rubric_files = _pick_matching_files(top_level_files, rubric_keywords)
    brief_files = _pick_matching_files(top_level_files, brief_keywords, exclude=rubric_files)
    marking_scheme_files = _pick_matching_files(top_level_files, marking_scheme_keywords, exclude=rubric_files + brief_files)
    graded_sample_files = _pick_matching_files(
        top_level_files,
        graded_sample_keywords,
        exclude=rubric_files + brief_files + marking_scheme_files,
    )
    other_files = _pick_matching_files(
        top_level_files,
        other_keywords,
        exclude=rubric_files + brief_files + marking_scheme_files + graded_sample_files,
    )

    child_submission_files = sorted(
        path
        for child in folder.iterdir()
        if child.is_dir()
        for path in child.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES
    )
    excluded_top_level = set(rubric_files + brief_files + marking_scheme_files + graded_sample_files + other_files)
    top_level_submissions = tuple(path for path in top_level_files if path not in excluded_top_level)
    submission_files = tuple(child_submission_files) + tuple(top_level_submissions)

    return AssessmentBundle(
        name=assessment_name or folder.name,
        folder=folder,
        submission_files=submission_files,
        rubric_files=rubric_files,
        brief_files=brief_files,
        marking_scheme_files=marking_scheme_files,
        graded_sample_files=graded_sample_files,
        other_files=other_files,
    )


def build_assessment_bundle(
    folder: Path,
    rubric_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["rubric"],
    brief_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["brief"],
    marking_scheme_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["marking_scheme"],
    graded_sample_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["graded_sample"],
    other_keywords: tuple[str, ...] = DEFAULT_CONTEXT_PATTERNS["other"],
) -> AssessmentBundle:
    files = _list_supported_files_recursive(folder)
    rubric_files = _pick_matching_files(files, rubric_keywords)
    brief_files = _pick_matching_files(files, brief_keywords, exclude=rubric_files)
    marking_scheme_files = _pick_matching_files(files, marking_scheme_keywords, exclude=rubric_files + brief_files)
    graded_sample_files = _pick_matching_files(
        files,
        graded_sample_keywords,
        exclude=rubric_files + brief_files + marking_scheme_files,
    )
    other_files = _pick_matching_files(
        files,
        other_keywords,
        exclude=rubric_files + brief_files + marking_scheme_files + graded_sample_files,
    )

    excluded = set(rubric_files + brief_files + marking_scheme_files + graded_sample_files + other_files)
    submission_files = tuple(path for path in files if path not in excluded)

    return AssessmentBundle(
        name=folder.name,
        folder=folder,
        submission_files=submission_files,
        rubric_files=rubric_files,
        brief_files=brief_files,
        marking_scheme_files=marking_scheme_files,
        graded_sample_files=graded_sample_files,
        other_files=other_files,
    )


def build_marking_context_from_bundle(bundle: AssessmentBundle) -> MarkingContext:
    return prepare_marking_context(
        rubric_text=read_paths_text(bundle.rubric_files),
        brief_text=read_paths_text(bundle.brief_files),
        marking_scheme_text=read_paths_text(bundle.marking_scheme_files),
        graded_sample_text=read_paths_text(bundle.graded_sample_files),
        other_context_text=read_paths_text(bundle.other_files),
    )


def apply_assessment_map_to_submission_parts(parts: list[SubmissionPart], assessment_map: AssessmentMap) -> list[SubmissionPart]:
    unit_by_key = {_normalize_label_key(unit.label): unit for unit in assessment_map.units}
    enriched: list[SubmissionPart] = []
    for part in parts:
        unit = unit_by_key.get(_normalize_label_key(part.label))
        if unit is None:
            enriched.append(part)
            continue
        enriched.append(
            SubmissionPart(
                label=part.label,
                focus_hint=part.focus_hint,
                anchor_text=part.anchor_text,
                section_text=part.section_text,
                max_mark=unit.max_mark if unit.max_mark is not None else part.max_mark,
                marking_guidance=unit.rubric_text or unit.marking_guidance or part.marking_guidance,
                question_text_exact=unit.question_text_exact or part.question_text_exact,
                task_type=unit.task_type or part.task_type,
            )
        )
    return enriched


def apply_prepared_artifact_to_submission_parts(
    parts: list[SubmissionPart],
    prepared_assessment_map: PreparedAssessmentMap,
) -> list[SubmissionPart]:
    prepared_unit_by_key = {
        _normalize_label_key(unit.label): unit
        for unit in prepared_assessment_map.prepared_units
    }
    enriched: list[SubmissionPart] = []
    for part in parts:
        prepared_unit = prepared_unit_by_key.get(_normalize_label_key(part.label))
        if prepared_unit is None or not prepared_unit.compact_criteria:
            enriched.append(part)
            continue
        compact_guidance = "\n".join(
            ["Compact scoring criteria:", *[f"- {criterion}" for criterion in prepared_unit.compact_criteria]]
        )
        merged_guidance = compact_guidance
        if part.marking_guidance.strip():
            merged_guidance = f"{compact_guidance}\n\nDetailed rubric:\n{part.marking_guidance.strip()}"
        enriched.append(
            SubmissionPart(
                label=part.label,
                focus_hint=part.focus_hint,
                anchor_text=part.anchor_text,
                section_text=part.section_text,
                max_mark=part.max_mark,
                marking_guidance=merged_guidance,
                question_text_exact=part.question_text_exact,
                task_type=part.task_type,
            )
        )
    return enriched


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


def _trim_support_text(text: str, max_chars: int = 1200) -> str:
    normalized = " ".join(text.split()).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _build_context_text(context: MarkingContext) -> str:
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
    return "\n\n".join(context_parts).strip()


def _build_part_support_text(context: MarkingContext, part: SubmissionPart) -> str:
    support_parts: list[str] = []
    structure_guidance = extract_structure_guidance(context)
    if structure_guidance:
        support_parts.append(f"STRUCTURE AND MARKS HINTS:\n{structure_guidance}")
    if context.brief_text.strip():
        support_parts.append(f"LOCAL ASSIGNMENT BRIEF SUPPORT:\n{_trim_support_text(context.brief_text)}")
    if context.marking_scheme_text.strip():
        support_parts.append(f"LOCAL MARKING SCHEME SUPPORT:\n{_trim_support_text(context.marking_scheme_text)}")
    if context.other_context_text.strip():
        support_parts.append(f"LOCAL OTHER SUPPORT:\n{_trim_support_text(context.other_context_text)}")
    if not support_parts and part.marking_guidance.strip():
        support_parts.append("Use only the section rubric and the section text.")
    return ("\n\n".join(support_parts) + "\n\n") if support_parts else ""


def _call_ollama_json(
    model_name: str,
    messages: list[dict[str, str]],
    ollama_url: str,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "format": "json",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }
    response = requests.post(ollama_url, json=payload, timeout=300)
    response.raise_for_status()
    response_json = response.json()
    content = extract_message_content(response_json)
    return parse_json_object(content)


def _run_final_result_with_retry(
    model_name: str,
    messages: list[dict[str, str]],
    expected_max_mark: float,
    diagnostics: SubmissionDiagnostics,
    parts: list[SubmissionPart],
    part_analyses: list[dict[str, Any]],
    ollama_url: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    current_messages = list(messages)
    for attempt in range(2):
        data = _call_ollama_json(model_name=model_name, messages=current_messages, ollama_url=ollama_url)
        try:
            return normalize_marking_result(
                data,
                expected_max_mark=expected_max_mark,
                diagnostics=diagnostics,
                parts=parts,
                part_analyses=part_analyses,
            )
        except ValueError as exc:
            last_error = exc
            if attempt == 1 or not _should_retry_final_result(exc):
                raise
            current_messages = _build_final_result_retry_messages(current_messages, exc)
    if last_error is not None:
        raise last_error
    raise ValueError("Final result generation failed without a captured error.")


def _should_retry_final_result(exc: ValueError) -> bool:
    message = str(exc)
    retryable_patterns = (
        "shorter than the required minimum length",
        "did not explicitly cover all detected sections",
    )
    return any(pattern in message for pattern in retryable_patterns)


def _should_retry_part_analysis(exc: ValueError) -> bool:
    message = str(exc)
    retryable_patterns = (
        "too few items for strengths",
        "too few items for weaknesses",
        "too few items for evidence",
        "did not return a coverage_comment",
    )
    return any(pattern in message for pattern in retryable_patterns)


def _normalize_grade_band(value: Any, score_pct: float, marking_guidance: str) -> str:
    text = str(value).strip().lower()
    analytical = any(band in marking_guidance.lower() for band in ("first", "2:1", "2:2", "third/pass", "missing/unfinished", "fail"))
    if analytical:
        mapping = {
            "first": "First",
            "1st": "First",
            "2:1": "2:1",
            "upper second": "2:1",
            "2:2": "2:2",
            "lower second": "2:2",
            "third": "Third/Pass",
            "pass": "Third/Pass",
            "third/pass": "Third/Pass",
            "fail": "Fail",
            "missing": "Missing/unfinished",
            "missing/unfinished": "Missing/unfinished",
            "unfinished": "Missing/unfinished",
        }
        if text in mapping:
            return mapping[text]
        if score_pct >= 70:
            return "First"
        if score_pct >= 60:
            return "2:1"
        if score_pct >= 50:
            return "2:2"
        if score_pct >= 40:
            return "Third/Pass"
        if score_pct <= 20:
            return "Missing/unfinished"
        return "Fail"

    mapping = {
        "excellent": "Excellent",
        "strong": "Strong",
        "secure": "Secure",
        "adequate": "Adequate",
        "developing": "Developing",
        "weak": "Weak",
        "missing": "Missing",
    }
    if text in mapping:
        return mapping[text]
    if score_pct >= 85:
        return "Excellent"
    if score_pct >= 70:
        return "Strong"
    if score_pct >= 55:
        return "Secure"
    if score_pct >= 40:
        return "Adequate"
    if score_pct >= 25:
        return "Developing"
    if score_pct > 0:
        return "Weak"
    return "Missing"


def _normalize_within_band_position(value: Any, score_pct: float, grade_band: str, marking_guidance: str) -> str:
    text = str(value).strip().lower()
    if text in {"low", "mid", "high"}:
        return text
    analytical = any(band in marking_guidance.lower() for band in ("first", "2:1", "2:2", "third/pass", "missing/unfinished", "fail"))
    if analytical:
        if grade_band == "First":
            return _position_from_bounds(score_pct, 70.0, 100.0)
        if grade_band == "2:1":
            return _position_from_bounds(score_pct, 60.0, 69.99)
        if grade_band == "2:2":
            return _position_from_bounds(score_pct, 50.0, 59.99)
        if grade_band == "Third/Pass":
            return _position_from_bounds(score_pct, 40.0, 49.99)
        if grade_band == "Fail":
            return _position_from_bounds(score_pct, 20.01, 39.99)
        return _position_from_bounds(score_pct, 0.0, 20.0)
    if grade_band == "Excellent":
        return _position_from_bounds(score_pct, 85.0, 100.0)
    if grade_band == "Strong":
        return _position_from_bounds(score_pct, 70.0, 84.99)
    if grade_band == "Secure":
        return _position_from_bounds(score_pct, 55.0, 69.99)
    if grade_band == "Adequate":
        return _position_from_bounds(score_pct, 40.0, 54.99)
    if grade_band == "Developing":
        return _position_from_bounds(score_pct, 25.0, 39.99)
    if grade_band == "Weak":
        return _position_from_bounds(score_pct, 0.01, 24.99)
    return "low"


def _position_from_bounds(score_pct: float, lower: float, upper: float) -> str:
    if upper <= lower:
        return "mid"
    width = upper - lower
    if score_pct < lower + (width / 3):
        return "low"
    if score_pct > upper - (width / 3):
        return "high"
    return "mid"


def _default_band_confidence(grade_band: str, marking_guidance: str) -> float:
    analytical = any(band in marking_guidance.lower() for band in ("first", "2:1", "2:2", "third/pass", "missing/unfinished", "fail"))
    if analytical and grade_band in {"Fail", "Missing/unfinished"}:
        return 78.0
    if analytical:
        return 72.0
    return 75.0


def _build_rubric_verification_retry_messages(
    messages: list[dict[str, str]],
    unit: AssessmentUnit,
    exc: Exception,
) -> list[dict[str, str]]:
    retry_note = (
        "Previous attempt failed. Return exactly one valid JSON object with keys "
        '"rubric_text", "issues", and "confidence_0_to_100". '
        f"Do not truncate the JSON for {unit.label}. "
        f"Failure reason: {exc}"
    )
    return [*messages, {"role": "user", "content": retry_note}]


def _build_final_result_retry_messages(messages: list[dict[str, str]], exc: ValueError) -> list[dict[str, str]]:
    message = str(exc)
    if "shorter than the required minimum length" in message:
        retry_note = (
            "Previous attempt failed because overall_feedback was too short. "
            "Retry now and make overall_feedback at least 180 words, concrete, and explicitly tied to the section analyses and section labels. "
            "Return only the required JSON object."
        )
    elif "did not explicitly cover all detected sections" in message:
        retry_note = (
            "Previous attempt failed because not all detected sections were explicitly covered. "
            "Retry now and mention every detected section label in overall_feedback and covered_parts. "
            "Return only the required JSON object."
        )
    else:
        retry_note = (
            f"Previous attempt failed validation: {message}. "
            "Retry now and satisfy the validation rule. Return only the required JSON object."
        )
    return [*messages, {"role": "user", "content": retry_note}]


def _build_part_analysis_retry_messages(
    messages: list[dict[str, str]],
    part: SubmissionPart,
    exc: ValueError,
) -> list[dict[str, str]]:
    retry_note = (
        f"Previous attempt for {part.label} failed validation: {exc}. "
        "Retry now and return at least 2 concrete strengths, at least 2 concrete weaknesses, at least 1 concrete evidence item, "
        "and a non-empty coverage_comment. Return only the required JSON object."
    )
    return [*messages, {"role": "user", "content": retry_note}]


def _normalize_feedback_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    content = value.get("content")
    if not isinstance(content, list):
        return ""
    lines: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        label = str(item.get("section_label", "")).strip()
        analysis = str(item.get("analysis", "")).strip()
        feedback = str(item.get("feedback", "")).strip()
        parts = [piece for piece in (label, analysis, feedback) if piece]
        if parts:
            lines.append(": ".join(parts[:1]) + (f" - {' '.join(parts[1:])}" if len(parts) > 1 else ""))
    return " ".join(lines).strip()


def _normalize_string_list(value: Any, field_name: str, label: str, minimum: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Model did not return a list for {field_name} in {label}: {value}")
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if len(normalized) < minimum:
        raise ValueError(f"Model returned too few items for {field_name} in {label}: {value}")
    return normalized


def _has_major_weaknesses(weaknesses: list[str], feedback: str) -> bool:
    text = " ".join(weaknesses + [feedback]).lower()
    patterns = (
        "incorrect",
        "missing",
        "does not address",
        "fails to",
        "not properly specified",
        "superficial",
        "underdeveloped",
        "limited discussion",
    )
    return any(pattern in text for pattern in patterns)


def _is_placeholder_section_label(label: str) -> bool:
    normalized = _normalize_label_key(label)
    placeholders = {
        "section above",
        "section below",
        "answer above",
        "answer below",
        "above",
        "below",
    }
    return normalized in placeholders


def _extract_text_from_docx_xml(raw: bytes) -> str:
    namespaces = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    }
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile):
        return ""

    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespaces):
        text_parts: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag in {"t", "delText"} and node.text:
                text_parts.append(node.text)
            elif tag == "tab":
                text_parts.append("\t")
        paragraph_text = "".join(text_parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)
    return "\n".join(paragraphs)


def _restore_label_from_key(key: str) -> str:
    parts = key.split()
    if len(parts) >= 2:
        return f"{parts[0].title()} {parts[1].upper() if parts[1].isalpha() else parts[1]}"
    return key.title()


def _segment_subparts_within_section(parent_part: SubmissionPart, child_specs: list[SubmissionPart]) -> list[SubmissionPart] | None:
    section_text = parent_part.section_text.strip()
    if not section_text:
        return None

    anchors: list[tuple[int, SubmissionPart]] = []
    used_positions: set[int] = set()
    for child in child_specs:
        position = _find_subpart_anchor_position(section_text, child)
        if position == -1 or position in used_positions:
            continue
        used_positions.add(position)
        anchors.append((position, child))

    if len(anchors) < max(2, len(child_specs) // 2 + 1):
        return None

    anchors.sort(key=lambda item: item[0])
    segmented: list[SubmissionPart] = []
    for index, (start, child) in enumerate(anchors):
        end = anchors[index + 1][0] if index + 1 < len(anchors) else len(section_text)
        child_text = section_text[start:end].strip()
        segmented.append(
            SubmissionPart(
                label=child.label,
                focus_hint=child.focus_hint,
                anchor_text=child.anchor_text,
                section_text=child_text,
                max_mark=child.max_mark,
                marking_guidance=child.marking_guidance,
                question_text_exact=child.question_text_exact,
                task_type=child.task_type,
            )
        )
    return segmented


def _resolve_expected_child_parts(
    parent_part: SubmissionPart,
    child_parts: list[SubmissionPart],
    child_specs: list[SubmissionPart],
) -> list[SubmissionPart]:
    child_by_key = {_normalize_label_key(child.label): child for child in child_parts}
    resolved: list[SubmissionPart] = []
    for expected in child_specs:
        matched = child_by_key.get(_normalize_label_key(expected.label))
        section_text = matched.section_text if matched is not None else parent_part.section_text
        resolved.append(
            SubmissionPart(
                label=expected.label,
                focus_hint=(matched.focus_hint if matched is not None else "") or expected.focus_hint,
                anchor_text=(matched.anchor_text if matched is not None else "") or expected.anchor_text,
                section_text=section_text,
                max_mark=expected.max_mark,
                marking_guidance=expected.marking_guidance,
                question_text_exact=expected.question_text_exact,
                task_type=expected.task_type,
            )
        )
    return resolved


def _find_subpart_anchor_position(section_text: str, child: SubmissionPart) -> int:
    normalized_text = section_text.replace("\r\n", "\n")
    question_match = re.search(r"\bQ(?P<id>\d+)\b", child.label, flags=re.IGNORECASE)
    if not question_match:
        return -1
    question_id = question_match.group("id")
    patterns = [
        rf"(?im)^\s*Q{question_id}\b",
        rf"(?im)^\s*Question\s+{question_id}\b",
        rf"(?im)^\s*{question_id}\.\s",
        rf"(?im)^\s*{question_id}\)\s",
        rf"(?im)^\s*\({question_id}\)\s",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_text)
        if match:
            return match.start()
    return -1


def _detect_subparts_with_model(
    parent_part: SubmissionPart,
    filename: str,
    child_specs: list[SubmissionPart],
    model_name: str,
    ollama_url: str,
) -> list[SubmissionPart] | None:
    data = _call_ollama_json(
        model_name=model_name,
        messages=build_subpart_structure_messages(parent_part, filename, child_specs),
        ollama_url=ollama_url,
    )
    detected = normalize_detected_parts(data)
    if len(detected) == 1 and _normalize_label_key(detected[0].label) == _normalize_label_key(parent_part.label):
        return None

    child_by_key = {_normalize_label_key(child.label): child for child in child_specs}
    normalized_children: list[SubmissionPart] = []
    for child in detected:
        key = _normalize_label_key(child.label)
        expected = child_by_key.get(key)
        if expected is None:
            continue
        normalized_children.append(
            SubmissionPart(
                label=expected.label,
                focus_hint=child.focus_hint or expected.focus_hint,
                anchor_text=child.anchor_text or expected.anchor_text,
                max_mark=expected.max_mark,
                marking_guidance=expected.marking_guidance,
                question_text_exact=expected.question_text_exact,
                task_type=expected.task_type,
            )
        )
    if len(normalized_children) < 2:
        return None
    return _segment_subparts_within_section(parent_part, normalized_children)


def _list_supported_files_recursive(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES)


def _pick_matching_files(
    files: list[Path],
    keywords: tuple[str, ...],
    exclude: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    excluded = set(exclude)
    if not keywords:
        return ()
    matches = []
    for path in files:
        if path in excluded:
            continue
        haystack = f"{path.stem} {path.parent.name}".lower().replace("_", " ")
        if any(keyword in haystack for keyword in keywords):
            matches.append(path)
    return tuple(matches)


def _normalize_label_key(label: str) -> str:
    return " ".join(label.lower().split())


def _find_part_anchor_position(normalized_text: str, part: SubmissionPart) -> int:
    candidates = [part.anchor_text.strip(), part.label.strip()]
    for candidate in candidates:
        if not candidate:
            continue
        exact = normalized_text.find(candidate)
        if exact != -1:
            return exact
        pattern = re.compile(re.escape(candidate), flags=re.IGNORECASE)
        match = pattern.search(normalized_text)
        if match:
            return match.start()
    return -1
