"""
Compute embedding features for every answer part.

For each answer part this module:
  - Embeds the cleaned_text
  - Embeds the rubric criteria text, model answer, and anchor answers (if available)
  - Computes cosine similarities against all of the above
  - Finds top-5 nearest neighbours within the same (question_id, part_id) group
  - Writes EmbeddingFeatures records to embedding_features.jsonl

CLI usage:
    python -m experiments.cross_sectional_part_marker.src.embeddings \\
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

from experiments.cross_sectional_part_marker.src.ollama_client import OllamaClient
from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerPart,
    EmbeddingFeatures,
    PipelineConfig,
    RubricSpec,
)

logger = logging.getLogger(__name__)


def answer_key(submission_id: str, question_id: str, part_id: str) -> str:
    """Stable key for one extracted answer part."""
    return f"{submission_id}::{question_id}::{part_id}"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_answer_parts(path: Path) -> list[AnswerPart]:
    parts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                parts.append(AnswerPart.model_validate_json(line))
    return parts


def _load_rubric_spec(path: Path) -> RubricSpec:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RubricSpec.model_validate(data)


def _rubric_criteria_text(rubric: RubricSpec, question_id: str, part_id: str) -> str:
    """Concatenate all criterion descriptions for a question part."""
    spec = rubric.get_part_spec(question_id, part_id)
    if spec is None:
        return ""
    return " ".join(c.name + ": " + c.description for c in spec.criteria)


def _top_k_neighbours(
    target_id: str,
    target_vec: list[float],
    group: dict[str, list[float]],
    k: int = 5,
) -> list[str]:
    """Return submission_ids of k nearest neighbours (excluding self)."""
    scored = [
        (sim_id, cosine_similarity(target_vec, vec))
        for sim_id, vec in group.items()
        if sim_id != target_id
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored[:k]]


def run_embeddings(config: PipelineConfig, force: bool = False) -> Path:
    """
    Compute and persist embedding features for all answer parts.
    """
    output_dir = Path(config.output_folder)
    output_path = output_dir / "embedding_features.jsonl"
    raw_vectors_path = output_dir / "raw_vectors.jsonl"

    if output_path.exists() and raw_vectors_path.exists() and not force:
        logger.info("embedding_features.jsonl and raw_vectors.jsonl exist — skipping (use --force to overwrite)")
        return output_path

    answer_parts_path = output_dir / "answer_parts.jsonl"
    rubric_path = output_dir / "rubric_spec.json"

    if not answer_parts_path.exists():
        raise FileNotFoundError(f"answer_parts.jsonl not found. Run ingest first.")
    if not rubric_path.exists():
        raise FileNotFoundError(f"rubric_spec.json not found. Run rubric_compiler first.")

    parts = _load_answer_parts(answer_parts_path)
    rubric = _load_rubric_spec(rubric_path)

    client = OllamaClient(
        base_url=config.ollama.base_url,
        timeout=config.ollama.timeout,
        stage="embeddings",
    )
    embed_model = config.models.embedding_model

    # --- Pre-compute rubric embeddings per (question_id, part_id) ---
    rubric_embeddings: dict[tuple[str, str], list[float]] = {}
    for qp in rubric.question_parts:
        key = (qp.question_id, qp.part_id)
        text = _rubric_criteria_text(rubric, qp.question_id, qp.part_id)
        if text.strip():
            try:
                rubric_embeddings[key] = client.embed(embed_model, text)
            except Exception as exc:
                logger.warning("Failed to embed rubric for %s%s: %s", qp.question_id, qp.part_id, exc)

    # --- Load sample/anchor answers if available ---
    # Anchor answers keyed by (question_id, part_id, bucket)
    anchor_embeddings: dict[tuple[str, str, str], list[float]] = {}
    if config.sample_answers_path:
        sample_path = Path(config.sample_answers_path)
        if sample_path.exists():
            try:
                with open(sample_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        q_id = sample.get("question_id", "")
                        p_id = sample.get("part_id", "")
                        bucket = sample.get("bucket", "")
                        text = sample.get("text", "")
                        if text and bucket in ("A", "B", "C"):
                            key = (q_id, p_id, bucket)
                            if key not in anchor_embeddings:
                                try:
                                    anchor_embeddings[key] = client.embed(embed_model, text)
                                except Exception as exc:
                                    logger.warning("Failed to embed anchor %s: %s", key, exc)
            except Exception as exc:
                logger.warning("Failed to load sample answers from %s: %s", sample_path, exc)

    # --- Embed all answer parts ---
    part_vectors: dict[str, list[float]] = {}  # answer_key -> vector
    for i, part in enumerate(parts):
        logger.info("Embedding %d/%d: %s %s%s", i + 1, len(parts), part.submission_id, part.question_id, part.part_id)
        text = part.cleaned_text or part.raw_text
        if not text.strip():
            logger.warning("Empty text for %s — skipping embedding", part.submission_id)
            continue
        try:
            vec = client.embed(embed_model, text)
            part_vectors[answer_key(part.submission_id, part.question_id, part.part_id)] = vec
        except Exception as exc:
            logger.error("Embed failed for %s: %s", part.submission_id, exc)

    with open(raw_vectors_path, "w", encoding="utf-8") as fh:
        for part in parts:
            key = answer_key(part.submission_id, part.question_id, part.part_id)
            vec = part_vectors.get(key)
            if vec is None:
                continue
            fh.write(json.dumps({
                "answer_key": key,
                "submission_id": part.submission_id,
                "question_id": part.question_id,
                "part_id": part.part_id,
                "embedding_model": embed_model,
                "vector": vec,
            }) + "\n")
    logger.info("Wrote raw vectors to %s", raw_vectors_path)

    # --- Group parts by (question_id, part_id) for NN search ---
    groups: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
    for part in parts:
        key = answer_key(part.submission_id, part.question_id, part.part_id)
        if key in part_vectors:
            groups[(part.question_id, part.part_id)][part.submission_id] = part_vectors[key]

    # --- Build EmbeddingFeatures records ---
    features: list[EmbeddingFeatures] = []
    for part in parts:
        vec = part_vectors.get(answer_key(part.submission_id, part.question_id, part.part_id))
        key = (part.question_id, part.part_id)

        sim_rubric: float | None = None
        if vec and key in rubric_embeddings:
            sim_rubric = round(cosine_similarity(vec, rubric_embeddings[key]), 4)

        sim_a: float | None = None
        sim_b: float | None = None
        sim_c: float | None = None
        if vec:
            k_a = (part.question_id, part.part_id, "A")
            k_b = (part.question_id, part.part_id, "B")
            k_c = (part.question_id, part.part_id, "C")
            if k_a in anchor_embeddings:
                sim_a = round(cosine_similarity(vec, anchor_embeddings[k_a]), 4)
            if k_b in anchor_embeddings:
                sim_b = round(cosine_similarity(vec, anchor_embeddings[k_b]), 4)
            if k_c in anchor_embeddings:
                sim_c = round(cosine_similarity(vec, anchor_embeddings[k_c]), 4)

        neighbours: list[str] = []
        if vec and key in groups:
            neighbours = _top_k_neighbours(part.submission_id, vec, groups[key], k=5)

        features.append(
            EmbeddingFeatures(
                submission_id=part.submission_id,
                question_id=part.question_id,
                part_id=part.part_id,
                embedding_model=embed_model,
                nearest_neighbours=neighbours,
                similarity_to_model_answer=None,  # model answer embedding not separately distinguished
                similarity_to_rubric=sim_rubric,
                similarity_to_anchor_A=sim_a,
                similarity_to_anchor_B=sim_b,
                similarity_to_anchor_C=sim_c,
                cluster_id=None,  # populated by clustering stage
                outlier_score=None,
            )
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        for f in features:
            fh.write(f.model_dump_json() + "\n")

    logger.info("Wrote %d embedding feature records to %s", len(features), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute embedding features for all answer parts")
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

    out = run_embeddings(cfg, force=args.force)
    print(f"Embeddings complete: {out}")
    sys.exit(0)
