"""
Export pipeline results into final deliverables:

  1. final_marks.csv
  2. detailed_feedback.xlsx
  3. moderation_report.html
  4. audit_trail.jsonl

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.export \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from experiments.cross_sectional_part_marker.src.audit import AuditLog
from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerPart,
    CalibratedAnswer,
    CrossSectionalStructure,
    Feedback,
    HumanReviewEntry,
    PipelineConfig,
)
from experiments.cross_sectional_part_marker.src.embeddings import answer_key

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path, model_cls):
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(model_cls.model_validate_json(line))
    return records


def _load_cross_sectional(path: Path) -> list[CrossSectionalStructure]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [CrossSectionalStructure.model_validate(d) for d in data]


def _review_map(review_path: Path) -> dict[str, HumanReviewEntry]:
    """Load human review log and build a key -> last entry map."""
    if not review_path.exists():
        return {}
    entries: list[HumanReviewEntry] = []
    with open(review_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(HumanReviewEntry.model_validate_json(line))
                except Exception:
                    pass
    # Keep last review per (submission_id, question_id, part_id)
    result: dict[str, HumanReviewEntry] = {}
    for e in entries:
        key = answer_key(e.submission_id, e.question_id, e.part_id)
        result[key] = e
    return result


def export_csv(
    calibrated: list[CalibratedAnswer],
    feedbacks: list[Feedback],
    reviews: dict[str, HumanReviewEntry],
    parts: dict[str, AnswerPart],
    output_path: Path,
) -> None:
    """Write final_marks.csv."""
    import csv

    feedback_map: dict[str, Feedback] = {
        answer_key(f.submission_id, f.question_id, f.part_id): f for f in feedbacks
    }

    with open(output_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "submission_id", "anonymised_student_id", "question_id", "part_id",
            "final_score", "max_score", "final_bucket", "feedback_summary", "review_status",
        ])
        writer.writeheader()
        for cal in calibrated:
            key = answer_key(cal.submission_id, cal.question_id, cal.part_id)
            fb = feedback_map.get(key)
            rev = reviews.get(key)
            part = parts.get(key)

            final_score = rev.human_score if (rev and rev.human_score is not None) else cal.calibrated_score
            final_bucket = rev.human_bucket if (rev and rev.human_bucket) else cal.calibrated_bucket
            feedback_summary = fb.feedback.summary if fb else ""
            review_status = rev.review_action if rev else ("needs_review" if cal.needs_human_review else "approved")
            max_score = cal.max_score or (fb.max_score if fb else 0.0)

            writer.writerow({
                "submission_id": cal.submission_id,
                "anonymised_student_id": cal.anonymised_student_id or (part.anonymised_student_id if part else ""),
                "question_id": cal.question_id,
                "part_id": cal.part_id,
                "final_score": final_score,
                "max_score": max_score,
                "final_bucket": final_bucket,
                "feedback_summary": feedback_summary,
                "review_status": review_status,
            })
    logger.info("Wrote %s", output_path)


def export_xlsx(
    calibrated: list[CalibratedAnswer],
    feedbacks: list[Feedback],
    reviews: dict[str, HumanReviewEntry],
    parts: dict[str, AnswerPart],
    output_path: Path,
) -> None:
    """Write detailed_feedback.xlsx."""
    try:
        import openpyxl
    except ImportError:
        logger.error("openpyxl not installed — cannot write .xlsx")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Detailed Feedback"

    headers = [
        "submission_id", "question_id", "part_id",
        "final_score", "max_score", "final_bucket",
        "initial_score", "calibrated_score", "initial_bucket", "calibrated_bucket",
        "calibration_notes", "nn_consistency", "confidence", "needs_review",
        "strengths", "limitations", "improvement_advice", "feedback_summary",
        "review_action", "reviewer", "review_notes",
    ]
    ws.append(headers)

    feedback_map: dict[str, Feedback] = {
        answer_key(f.submission_id, f.question_id, f.part_id): f for f in feedbacks
    }

    for cal in calibrated:
        key = answer_key(cal.submission_id, cal.question_id, cal.part_id)
        fb = feedback_map.get(key)
        rev = reviews.get(key)
        final_score = rev.human_score if (rev and rev.human_score is not None) else cal.calibrated_score
        final_bucket = rev.human_bucket if (rev and rev.human_bucket) else cal.calibrated_bucket
        max_score = cal.max_score or (fb.max_score if fb else 0.0)
        ws.append([
            cal.submission_id, cal.question_id, cal.part_id,
            final_score, max_score,
            final_bucket,
            cal.initial_score, cal.calibrated_score, cal.initial_bucket, cal.calibrated_bucket,
            cal.calibration_notes, cal.nearest_neighbour_consistency,
            cal.confidence, cal.needs_human_review,
            fb.feedback.strengths if fb else "",
            fb.feedback.limitations if fb else "",
            fb.feedback.improvement_advice if fb else "",
            fb.feedback.summary if fb else "",
            rev.review_action if rev else "",
            rev.reviewer if rev else "",
            rev.review_notes if rev else "",
        ])

    wb.save(output_path)
    logger.info("Wrote %s", output_path)


def export_html_report(
    calibrated: list[CalibratedAnswer],
    feedbacks: list[Feedback],
    structures: list[CrossSectionalStructure],
    reviews: dict[str, HumanReviewEntry],
    output_path: Path,
) -> None:
    """Write moderation_report.html."""
    from collections import Counter

    bucket_counts: Counter = Counter(c.calibrated_bucket for c in calibrated)
    n_total = len(calibrated)
    n_flagged = sum(1 for c in calibrated if c.needs_human_review)
    n_reviewed = len(reviews)

    scores = [c.calibrated_score for c in calibrated]
    mean_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score_val = max(scores) if scores else 0

    bucket_rows = "".join(
        f"<tr><td>{b}</td><td>{bucket_counts.get(b, 0)}</td></tr>"
        for b in ["A", "B", "C", "D", "E", "F"]
    )

    cluster_sections = ""
    for struct in structures:
        cluster_rows = "".join(
            f"<tr><td>{cl.cluster_id}</td><td>{cl.size}</td><td>{cl.likely_quality_band}</td>"
            f"<td>{cl.summary}</td></tr>"
            for cl in struct.clusters
        )
        cluster_sections += (
            f"<h3>{struct.question_id}{struct.part_id} — {struct.n_answers} answers, "
            f"{len(struct.outliers)} outliers</h3>"
            f"<table border='1'><tr><th>Cluster</th><th>Size</th><th>Quality Band</th><th>Summary</th></tr>"
            f"{cluster_rows}</table>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Moderation Report</title>
<style>body{{font-family:Arial,sans-serif;margin:40px}}table{{border-collapse:collapse;margin-bottom:20px}}
th,td{{padding:8px 12px;border:1px solid #ccc}}th{{background:#f0f0f0}}</style>
</head>
<body>
<h1>Moderation Report</h1>
<h2>Summary Statistics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total answers</td><td>{n_total}</td></tr>
<tr><td>Mean score</td><td>{mean_score:.2f}</td></tr>
<tr><td>Min score</td><td>{min_score:.2f}</td></tr>
<tr><td>Highest final score</td><td>{max_score_val:.2f}</td></tr>
<tr><td>Flagged for review</td><td>{n_flagged} ({100*n_flagged//n_total if n_total else 0}%)</td></tr>
<tr><td>Human-reviewed</td><td>{n_reviewed}</td></tr>
</table>

<h2>Bucket Distribution</h2>
<table>
<tr><th>Bucket</th><th>Count</th></tr>
{bucket_rows}
</table>

<h2>Cluster Summaries</h2>
{cluster_sections if cluster_sections else "<p>No clustering data available.</p>"}
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", output_path)


def run_export(config: PipelineConfig, force: bool = False) -> dict[str, Path]:
    """
    Export all outputs to final deliverables.

    Returns dict of output name -> path.
    """
    output_dir = Path(config.output_folder)

    calibrated_path = output_dir / "calibrated_scores.jsonl"
    feedback_path = output_dir / "feedback.jsonl"
    structure_path = output_dir / "cross_sectional_structure.json"
    review_log_path = output_dir / "human_review_log.jsonl"
    answer_parts_path = output_dir / "answer_parts.jsonl"

    if not calibrated_path.exists():
        raise FileNotFoundError("calibrated_scores.jsonl not found. Run calibration first.")

    calibrated: list[CalibratedAnswer] = _load_jsonl(calibrated_path, CalibratedAnswer)
    feedbacks: list[Feedback] = _load_jsonl(feedback_path, Feedback)
    structures = _load_cross_sectional(structure_path)
    reviews = _review_map(review_log_path)
    parts_list: list[AnswerPart] = _load_jsonl(answer_parts_path, AnswerPart)
    parts: dict[str, AnswerPart] = {
        answer_key(p.submission_id, p.question_id, p.part_id): p for p in parts_list
    }

    outputs: dict[str, Path] = {}

    # 1. final_marks.csv
    csv_path = output_dir / "final_marks.csv"
    if not csv_path.exists() or force:
        export_csv(calibrated, feedbacks, reviews, parts, csv_path)
    outputs["final_marks_csv"] = csv_path

    # 2. detailed_feedback.xlsx
    xlsx_path = output_dir / "detailed_feedback.xlsx"
    if not xlsx_path.exists() or force:
        export_xlsx(calibrated, feedbacks, reviews, parts, xlsx_path)
    outputs["detailed_feedback_xlsx"] = xlsx_path

    # 3. moderation_report.html
    html_path = output_dir / "moderation_report.html"
    if not html_path.exists() or force:
        export_html_report(calibrated, feedbacks, structures, reviews, html_path)
    outputs["moderation_report_html"] = html_path

    # 4. audit_trail.jsonl
    audit_path = output_dir / "audit_trail.jsonl"
    AuditLog.instance().save(audit_path)
    outputs["audit_trail_jsonl"] = audit_path

    logger.info("Export complete: %d output files", len(outputs))
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export pipeline results to final deliverables")
    p.add_argument("--config", required=True, help="Path to YAML pipeline config")
    p.add_argument("--force", action="store_true", help="Overwrite existing exports")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    import sys

    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    cfg = PipelineConfig.model_validate(raw_cfg)

    result = run_export(cfg, force=args.force)
    for name, path in result.items():
        print(f"  {name}: {path}")
    sys.exit(0)
