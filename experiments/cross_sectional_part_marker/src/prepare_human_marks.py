"""
Parse EC3040 Moodle grades Excel and produce human_marks.csv for validation.

Reads per-question breakdowns from feedback comments and aggregates to
composite Part scores (Part_2 = Q1-Q6 sum, Part_3 = Q1-Q4 sum).

Output columns: anonymised_student_id, question_id, part_id, human_score

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.prepare_human_marks \\
        --input codex/marking_test/input/EC3040_deanonymised_grades.xlsx \\
        --output experiments/cross_sectional_part_marker/outputs/human_marks.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


_BREAKDOWN_RE = re.compile(
    r"Part\s+(\d+)(?:\s+Q(\d+))?:\s+([\d.]+)/[\d.]+"
)
_PARTICIPANT_ID_RE = re.compile(r"(?<!\d)(\d{7})(?!\d)")


def parse_feedback(text: str) -> dict[tuple[str, str], float]:
    """Return {(question_id, part_id): score} from a feedback string."""
    scores: dict[tuple[str, str], float] = {}
    for m in _BREAKDOWN_RE.finditer(text):
        part_num = m.group(1)
        q_num = m.group(2)
        score = float(m.group(3))
        if q_num:
            question_id = f"Part_{part_num}"
            part_id = f"Q{q_num}"
        else:
            question_id = f"Part_{part_num}"
            part_id = ""
        scores[(question_id, part_id)] = score
    return scores


def aggregate_composites(
    raw: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """
    Compute composite Part_2 and Part_3 totals from sub-question scores.
    Also keeps Part_1 and Part_4 as-is.
    Returns only the four composite entries the pipeline scored.
    """
    result: dict[tuple[str, str], float] = {}

    for part in ("Part_1", "Part_4"):
        val = raw.get((part, ""))
        if val is not None:
            result[(part, "")] = val

    # Part_2 composite: Q1-Q6
    p2_subs = [raw.get(("Part_2", f"Q{i}"), 0.0) for i in range(1, 7)]
    # Use explicit Part_2 total if available (some rows may have it), else sum
    p2_total = raw.get(("Part_2", ""))
    if p2_total is not None:
        result[("Part_2", "")] = p2_total
    elif any(v > 0 for v in p2_subs):
        result[("Part_2", "")] = sum(p2_subs)

    # Part_3 composite: Q1-Q4
    p3_subs = [raw.get(("Part_3", f"Q{i}"), 0.0) for i in range(1, 5)]
    p3_total = raw.get(("Part_3", ""))
    if p3_total is not None:
        result[("Part_3", "")] = p3_total
    elif any(v > 0 for v in p3_subs):
        result[("Part_3", "")] = sum(p3_subs)

    return result


def run(input_path: Path, output_path: Path) -> None:
    try:
        import openpyxl
    except ImportError:
        sys.exit("openpyxl is required: pip install openpyxl")

    wb = openpyxl.load_workbook(str(input_path))
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    header = [str(c).strip() if c else "" for c in rows[0]]
    id_col = header.index("Identifier")
    feedback_col = header.index("Feedback comments")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["anonymised_student_id", "question_id", "part_id", "human_score"])

        for row in rows[1:]:
            identifier = str(row[id_col]) if row[id_col] else ""
            feedback = str(row[feedback_col]) if row[feedback_col] else ""

            id_match = _PARTICIPANT_ID_RE.search(identifier)
            if not id_match:
                skipped += 1
                continue
            anon_id = f"ANON_{id_match.group(1)}"

            raw_scores = parse_feedback(feedback)
            if not raw_scores:
                skipped += 1
                continue

            composite = aggregate_composites(raw_scores)
            for (qid, pid), score in sorted(composite.items()):
                writer.writerow([anon_id, qid, pid, score])
                written += 1

    print(f"Wrote {written} rows ({written // 4} students) to {output_path}")
    if skipped:
        print(f"Skipped {skipped} rows (no participant ID or feedback)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert Moodle grades Excel to human_marks.csv")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    run(args.input, args.output)
