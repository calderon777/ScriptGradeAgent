"""
Validate AI marks against human marks (if available) and run fairness checks.

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.validation \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml \\
        [--human-marks path/to/human_marks.csv]

Outputs:
    <output_folder>/validation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.cross_sectional_part_marker.src.schemas import (
    CalibratedAnswer,
    EmbeddingFeatures,
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


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den > 0 else float("nan")


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    def _rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        for rank, (orig_idx, _) in enumerate(sorted_vals, 1):
            ranks[orig_idx] = float(rank)
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    return _pearson_r(rx, ry)


def _mae(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def _median_ae(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    return statistics.median(abs(x - y) for x, y in zip(xs, ys))


def _qwk(xs: list[int], ys: list[int]) -> float:
    """Quadratic weighted kappa (integer labels)."""
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore
        return float(cohen_kappa_score(xs, ys, weights="quadratic"))
    except ImportError:
        logger.warning("sklearn not available — QWK not computed")
        return float("nan")
    except Exception as exc:
        logger.warning("QWK computation failed: %s", exc)
        return float("nan")


def _bucket_agreement(ai_buckets: list[str], human_buckets: list[str]) -> dict[str, float]:
    _ORDER = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    exact = sum(1 for a, h in zip(ai_buckets, human_buckets) if a == h)
    adjacent = sum(
        1 for a, h in zip(ai_buckets, human_buckets)
        if abs(_ORDER.get(a, -1) - _ORDER.get(h, -1)) <= 1
    )
    n = len(ai_buckets)
    return {
        "exact_match_rate": round(exact / n, 4) if n else 0.0,
        "adjacent_match_rate": round(adjacent / n, 4) if n else 0.0,
    }


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def run_validation(
    config: PipelineConfig,
    human_marks_csv: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Validate AI marks and produce validation_report.json.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "validation_report.json"

    if output_path.exists() and not force:
        logger.info("validation_report.json exists — skipping (use --force to overwrite)")
        return output_path

    calibrated_path = output_dir / "calibrated_scores.jsonl"
    embedding_path = output_dir / "embedding_features.jsonl"

    if not calibrated_path.exists():
        raise FileNotFoundError("calibrated_scores.jsonl not found. Run calibration first.")

    calibrated: list[CalibratedAnswer] = _load_jsonl(calibrated_path, CalibratedAnswer)
    embedding_feats: list[EmbeddingFeatures] = _load_jsonl(embedding_path, EmbeddingFeatures)

    report: dict[str, Any] = {
        "n_answers": len(calibrated),
        "n_flagged_for_review": sum(1 for c in calibrated if c.needs_human_review),
        "per_question_part": {},
        "consistency_warnings": [],
        "fairness_checks": {},
    }

    # --- Human marks comparison ---
    if human_marks_csv and human_marks_csv.exists():
        import csv

        logger.info("Loading human marks from %s", human_marks_csv)
        # Key: anonymised_student_id::question_id::part_id
        human_map: dict[str, float] = {}
        human_bucket_map: dict[str, str] = {}
        with open(human_marks_csv, newline="", encoding="utf-8") as csvf:
            reader = csv.DictReader(csvf)
            for row in reader:
                # Support both submission_id and anonymised_student_id columns
                sid = row.get("anonymised_student_id") or row.get("submission_id", "")
                key = answer_key(sid, row["question_id"], row["part_id"])
                human_map[key] = float(row["human_score"])
                if "human_bucket" in row:
                    human_bucket_map[key] = row["human_bucket"]

        def _cal_key(c: CalibratedAnswer) -> str:
            return answer_key(c.anonymised_student_id, c.question_id, c.part_id)

        # Overall metrics
        pairs = [
            (c.calibrated_score, human_map[_cal_key(c)])
            for c in calibrated
            if _cal_key(c) in human_map
        ]
        if pairs:
            ai_scores, h_scores = zip(*pairs)
            ai_scores = list(ai_scores)
            h_scores = list(h_scores)
            report["overall"] = {
                "n_matched": len(pairs),
                "mae": round(_mae(ai_scores, h_scores), 4),
                "median_ae": round(_median_ae(ai_scores, h_scores), 4),
                "pearson_r": round(_pearson_r(ai_scores, h_scores), 4),
                "spearman_rho": round(_spearman_rho(ai_scores, h_scores), 4),
                "qwk": round(_qwk([round(s) for s in ai_scores], [round(s) for s in h_scores]), 4),
            }
        else:
            logger.warning("No calibrated answers matched human marks — check anonymised_student_id alignment")

        # Per-question-part metrics
        by_qp: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        for c in calibrated:
            key = _cal_key(c)
            if key in human_map:
                by_qp[(c.question_id, c.part_id)].append((c.calibrated_score, human_map[key]))

        for (qid, pid), qp_pairs in by_qp.items():
            ai_qp, h_qp = zip(*qp_pairs)
            ai_qp, h_qp = list(ai_qp), list(h_qp)
            report["per_question_part"][f"{qid}_{pid}"] = {
                "n": len(qp_pairs),
                "mae": round(_mae(ai_qp, h_qp), 4),
                "pearson_r": round(_pearson_r(ai_qp, h_qp), 4),
            }

        # Bucket agreement
        ai_buckets = [
            c.calibrated_bucket for c in calibrated
            if _cal_key(c) in human_bucket_map
        ]
        h_buckets = [
            human_bucket_map[_cal_key(c)]
            for c in calibrated
            if _cal_key(c) in human_bucket_map
        ]
        if ai_buckets:
            report["bucket_agreement"] = _bucket_agreement(ai_buckets, h_buckets)

    else:
        logger.info("No human marks provided — skipping accuracy metrics")
        report["overall"] = {"note": "No human marks provided"}

    # --- Nearest-neighbour consistency checks ---
    feat_map: dict[str, EmbeddingFeatures] = {
        answer_key(f.submission_id, f.question_id, f.part_id): f for f in embedding_feats
    }
    score_map: dict[str, float] = {
        answer_key(c.submission_id, c.question_id, c.part_id): c.calibrated_score for c in calibrated
    }

    consistency_warnings: list[dict] = []
    checked_pairs: set[frozenset] = set()
    for c in calibrated:
        key = answer_key(c.submission_id, c.question_id, c.part_id)
        feat = feat_map.get(key)
        if not feat:
            continue
        for nn_id in feat.nearest_neighbours:
            nn_key = answer_key(nn_id, c.question_id, c.part_id)
            pair = frozenset([key, nn_key])
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            nn_score = score_map.get(nn_key)
            if nn_score is None:
                continue
            discrepancy = abs(c.calibrated_score - nn_score)
            if discrepancy > 1.0:
                consistency_warnings.append({
                    "submission_a": c.submission_id,
                    "submission_b": nn_id,
                    "question_id": c.question_id,
                    "part_id": c.part_id,
                    "score_a": c.calibrated_score,
                    "score_b": nn_score,
                    "discrepancy": round(discrepancy, 3),
                })

    report["consistency_warnings"] = consistency_warnings
    report["n_consistency_warnings"] = len(consistency_warnings)
    if consistency_warnings:
        logger.warning("%d nearest-neighbour consistency warnings found", len(consistency_warnings))

    # --- Fairness: similar answers with score discrepancy > 1 ---
    # (Re-uses the consistency_warnings list — similarity is implicit from NN structure)
    report["fairness_checks"] = {
        "method": "nearest-neighbour-based (cosine sim > 0 implied by NN membership)",
        "n_discrepant_nn_pairs": len(consistency_warnings),
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    logger.info("Validation report written to %s", output_path)

    # Print summary table
    if "overall" in report and "mae" in report["overall"]:
        ov = report["overall"]
        print(f"\n{'='*50}")
        print("VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"  N matched:         {ov.get('n_matched', '?')}")
        print(f"  MAE:               {ov.get('mae', '?'):.4f}")
        print(f"  Median AE:         {ov.get('median_ae', '?'):.4f}")
        print(f"  Pearson r:         {ov.get('pearson_r', '?'):.4f}")
        print(f"  Spearman rho:      {ov.get('spearman_rho', '?'):.4f}")
        print(f"  QWK:               {ov.get('qwk', '?'):.4f}")
        print(f"  NN warnings:       {len(consistency_warnings)}")
        print(f"{'='*50}\n")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate AI marks against human marks")
    p.add_argument("--config", required=True, help="Path to YAML pipeline config")
    p.add_argument("--human-marks", type=Path, default=None, help="CSV with human marks (optional)")
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

    out = run_validation(cfg, human_marks_csv=args.human_marks, force=args.force)
    print(f"Validation complete: {out}")
    sys.exit(0)
