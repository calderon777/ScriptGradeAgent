"""
Cross-sectional clustering of answer embeddings.

For each (question_id, part_id) group:
  - Loads embeddings from embedding_features.jsonl
  - Runs HDBSCAN (preferred) or KMeans
  - Optionally computes 2D UMAP coordinates
  - Identifies outliers, borderline cases, representative answers
  - Writes cross_sectional_structure.json

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.clustering \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.schemas import (
    ClusterSummary,
    CrossSectionalStructure,
    EmbeddingFeatures,
    PipelineConfig,
)
from experiments.cross_sectional_part_marker.src.embeddings import answer_key

logger = logging.getLogger(__name__)

# Fraction of answers to treat as outlier candidates (for KMeans fallback)
_OUTLIER_FRACTION = 0.1
# Fraction near cluster boundary to flag as borderline
_BORDERLINE_FRACTION = 0.15


def _load_embedding_features(path: Path) -> list[EmbeddingFeatures]:
    features = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                features.append(EmbeddingFeatures.model_validate_json(line))
    return features


def _save_embedding_features(features: list[EmbeddingFeatures], path: Path) -> None:
    """Overwrite embedding_features.jsonl with updated cluster_id and outlier_score."""
    with open(path, "w", encoding="utf-8") as fh:
        for f in features:
            fh.write(f.model_dump_json() + "\n")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]


def _cluster_numpy(
    ids: list[str],
    vectors: list[list[float]],
    run_umap: bool,
) -> tuple[list[int], list[float], list[list[float]] | None]:
    """
    Run HDBSCAN (preferred) or KMeans on vectors.

    Returns (labels, outlier_scores, umap_coords_or_None).
    Outlier scores: 0 = not outlier, higher = more outlier-like.
    """
    try:
        import numpy as np

        X = np.array(vectors, dtype=float)
    except ImportError:
        logger.warning("numpy not available — cannot run clustering")
        return [0] * len(ids), [0.0] * len(ids), None

    labels: list[int]
    outlier_scores: list[float]

    # Try HDBSCAN
    try:
        import hdbscan  # type: ignore

        min_cluster_size = max(2, len(ids) // 10)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
        clusterer.fit(X)
        labels = clusterer.labels_.tolist()
        outlier_scores = clusterer.outlier_scores_.tolist() if hasattr(clusterer, "outlier_scores_") else [0.0] * len(ids)
        logger.info("HDBSCAN: %d clusters, %d noise points", max(labels) + 1, labels.count(-1))
    except ImportError:
        logger.info("hdbscan not available — falling back to KMeans")
        from sklearn.cluster import KMeans  # type: ignore

        n_clusters = max(2, min(8, len(ids) // 5))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km.fit(X)
        labels = km.labels_.tolist()
        # Compute outlier score as distance from centroid (normalised)
        dists = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
        max_d = dists.max() if dists.max() > 0 else 1.0
        outlier_scores = (dists / max_d).tolist()

    # Optionally run UMAP
    umap_coords: list[list[float]] | None = None
    if run_umap and len(ids) >= 4:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(X)
            umap_coords = coords_2d.tolist()
            logger.info("UMAP projection computed (%d points)", len(ids))
        except ImportError:
            logger.info("umap-learn not available — skipping UMAP")
        except Exception as exc:
            logger.warning("UMAP failed: %s — skipping", exc)

    return labels, outlier_scores, umap_coords


def _find_medoid(
    member_ids: list[str],
    id_to_vec: dict[str, list[float]],
) -> str:
    """Return the submission_id closest to the centroid (medoid)."""
    vecs = [id_to_vec[sid] for sid in member_ids if sid in id_to_vec]
    if not vecs:
        return member_ids[0]
    c = _centroid(vecs)
    best_id = member_ids[0]
    best_dist = float("inf")
    for sid in member_ids:
        if sid not in id_to_vec:
            continue
        d = _euclidean(id_to_vec[sid], c)
        if d < best_dist:
            best_dist = d
            best_id = sid
    return best_id


def _cluster_group(
    ids: list[str],
    id_to_vec: dict[str, list[float]],
    run_umap: bool,
) -> tuple[list[int], list[float], list[list[float]] | None]:
    """Cluster a single (question_id, part_id) group."""
    vectors = [id_to_vec.get(sid, []) for sid in ids]
    # Filter out entries with no vector
    valid_ids = [sid for sid, v in zip(ids, vectors) if v]
    valid_vecs = [v for v in vectors if v]

    if len(valid_ids) < 2:
        return [0] * len(ids), [0.0] * len(ids), None

    labels_valid, scores_valid, umap_valid = _cluster_numpy(valid_ids, valid_vecs, run_umap)

    # Map back to original order
    valid_map: dict[str, tuple[int, float]] = {
        sid: (labels_valid[i], scores_valid[i]) for i, sid in enumerate(valid_ids)
    }
    labels = [valid_map.get(sid, (-1, 0.0))[0] for sid in ids]
    scores = [valid_map.get(sid, (-1, 0.0))[1] for sid in ids]

    return labels, scores, umap_valid


def _identify_borderlines(
    ids: list[str],
    labels: list[int],
    id_to_vec: dict[str, list[float]],
) -> list[str]:
    """
    Identify answers near cluster boundaries.
    Strategy: for each answer, compute similarity to nearest *different* cluster centroid.
    If that similarity is > 0.85 of within-cluster similarity, flag as borderline.
    """
    cluster_members: dict[int, list[str]] = defaultdict(list)
    for sid, label in zip(ids, labels):
        if label >= 0:
            cluster_members[label].append(sid)

    centroids: dict[int, list[float]] = {}
    for label, members in cluster_members.items():
        vecs = [id_to_vec[sid] for sid in members if sid in id_to_vec]
        if vecs:
            centroids[label] = _centroid(vecs)

    borderline = []
    for sid, label in zip(ids, labels):
        if label < 0 or sid not in id_to_vec:
            continue
        vec = id_to_vec[sid]
        within_sim = _cosine_sim(vec, centroids[label]) if label in centroids else 0.0
        other_sims = [
            _cosine_sim(vec, centroid)
            for lbl, centroid in centroids.items()
            if lbl != label
        ]
        if other_sims:
            nearest_other = max(other_sims)
            if nearest_other > 0.85 * within_sim:
                borderline.append(sid)
    return borderline


def run_clustering(config: PipelineConfig, force: bool = False) -> Path:
    """
    Run cross-sectional clustering on all embedding features.

    Returns path to cross_sectional_structure.json.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "cross_sectional_structure.json"
    embedding_path = output_dir / "embedding_features.jsonl"

    if output_path.exists() and not force:
        logger.info("cross_sectional_structure.json exists — skipping (use --force to overwrite)")
        return output_path

    if not embedding_path.exists():
        raise FileNotFoundError(f"embedding_features.jsonl not found. Run embeddings first.")

    all_features = _load_embedding_features(embedding_path)
    if not all_features:
        logger.warning("No embedding features found — nothing to cluster")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([], fh)
        return output_path

    # Build vector lookup from the JSONL; features don't store raw vectors —
    # we re-load from a separate raw vectors file if it exists, otherwise
    # clustering operates on the cosine similarity columns only.
    # Since we don't persist raw vectors in EmbeddingFeatures, we use
    # nearest-neighbour structure as a proxy for grouping.
    # If numpy/sklearn are available we attempt proper vector clustering
    # via a re-embedding step; otherwise we fall back to NN-graph clustering.
    raw_vectors_path = output_dir / "raw_vectors.jsonl"
    raw_vectors: dict[str, list[float]] = {}
    if raw_vectors_path.exists():
        with open(raw_vectors_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("answer_key") or answer_key(
                    rec["submission_id"],
                    rec.get("question_id", ""),
                    rec.get("part_id", ""),
                )
                raw_vectors[key] = rec["vector"]

    # Group features by (question_id, part_id)
    groups: dict[tuple[str, str], list[EmbeddingFeatures]] = defaultdict(list)
    for feat in all_features:
        groups[(feat.question_id, feat.part_id)].append(feat)

    structures: list[CrossSectionalStructure] = []
    # Also update embedding features with cluster_id and outlier_score
    feat_by_key: dict[str, EmbeddingFeatures] = {
        answer_key(f.submission_id, f.question_id, f.part_id): f for f in all_features
    }

    for (question_id, part_id), group_feats in groups.items():
        ids = [f.submission_id for f in group_feats]
        local_vectors = {
            f.submission_id: raw_vectors[answer_key(f.submission_id, f.question_id, f.part_id)]
            for f in group_feats
            if answer_key(f.submission_id, f.question_id, f.part_id) in raw_vectors
        }
        n = len(ids)
        logger.info("Clustering %s%s: %d answers", question_id, part_id, n)

        if n < 2:
            # Single answer — trivial cluster
            cluster = ClusterSummary(
                cluster_id=0,
                size=n,
                representative_submission_ids=ids[:1],
                summary="Single answer — no clustering performed",
                likely_quality_band="unknown",
            )
            structures.append(
                CrossSectionalStructure(
                    question_id=question_id,
                    part_id=part_id,
                    n_answers=n,
                    clusters=[cluster],
                    outliers=[],
                    borderline_cases=[],
                )
            )
            continue

        if local_vectors:
            labels, outlier_scores, _ = _cluster_group(
                ids, local_vectors, config.run_umap
            )
        else:
            # No raw vectors — assign everyone to cluster 0
            logger.warning(
                "No raw_vectors.jsonl found for %s%s — assigning all to cluster 0. "
                "Re-run embeddings to enable proper clustering.",
                question_id, part_id,
            )
            labels = [0] * n
            outlier_scores = [0.0] * n

        # Update EmbeddingFeatures in-memory
        for feat, label, score in zip(group_feats, labels, outlier_scores):
            key = answer_key(feat.submission_id, feat.question_id, feat.part_id)
            if key in feat_by_key:
                feat_by_key[key] = feat_by_key[key].model_copy(
                    update={"cluster_id": label, "outlier_score": round(score, 4)}
                )

        # Identify outliers (HDBSCAN noise = -1, or top fraction)
        outlier_threshold = sorted(outlier_scores, reverse=True)[
            max(0, int(n * _OUTLIER_FRACTION) - 1)
        ]
        outliers = [
            sid for sid, lbl, sc in zip(ids, labels, outlier_scores)
            if lbl == -1 or sc >= outlier_threshold
        ]

        # Identify borderlines
        if local_vectors:
            borderlines = _identify_borderlines(ids, labels, local_vectors)
        else:
            borderlines = []

        # Build ClusterSummary per cluster (exclude noise=-1)
        cluster_ids_set = sorted(set(l for l in labels if l >= 0))
        cluster_summaries: list[ClusterSummary] = []
        for cid in cluster_ids_set:
            members = [sid for sid, lbl in zip(ids, labels) if lbl == cid]
            rep = [_find_medoid(members, local_vectors)] if local_vectors and members else members[:1]
            cluster_summaries.append(
                ClusterSummary(
                    cluster_id=cid,
                    size=len(members),
                    representative_submission_ids=rep,
                    summary=f"Cluster {cid} — {len(members)} answers",
                    likely_quality_band="",
                )
            )

        structures.append(
            CrossSectionalStructure(
                question_id=question_id,
                part_id=part_id,
                n_answers=n,
                clusters=cluster_summaries,
                outliers=outliers,
                borderline_cases=borderlines,
            )
        )

    # Write cross_sectional_structure.json
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump([s.model_dump() for s in structures], fh, indent=2)
    logger.info("Wrote clustering results to %s", output_path)

    # Update embedding_features.jsonl with cluster_id / outlier_score
    updated_features = [
        feat_by_key[answer_key(f.submission_id, f.question_id, f.part_id)]
        for f in all_features
    ]
    _save_embedding_features(updated_features, embedding_path)
    logger.info("Updated embedding_features.jsonl with cluster assignments")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run cross-sectional clustering on answer embeddings")
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

    out = run_clustering(cfg, force=args.force)
    print(f"Clustering complete: {out}")
    sys.exit(0)
