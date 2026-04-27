"""
Master CLI entry point — runs all pipeline stages in order.

Usage:
    python -m experiments.cross_sectional_part_marker.src.run_pipeline \\
        --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml \\
        [--stages ingest,rubric,analyse,embed,cluster,score,calibrate,feedback,validate,export] \\
        [--force]

Available stages (in order):
    ingest      — extract answer parts from submission files
    rubric      — compile rubric_spec.json
    analyse     — structural analysis of each answer part
    embed       — compute embedding features
    cluster     — cross-sectional clustering
    score       — score each answer against rubric criteria
    calibrate   — pairwise calibration of scores
    feedback    — generate student feedback
    validate    — validation report (optional --human-marks)
    export      — produce final deliverables
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

from experiments.cross_sectional_part_marker.src.schemas import PipelineConfig

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    "ingest",
    "rubric",
    "analyse",
    "embed",
    "cluster",
    "score",
    "calibrate",
    "feedback",
    "validate",
    "export",
]


def _run_stage(name: str, config: PipelineConfig, force: bool) -> Path | dict | None:
    """Dispatch to the appropriate stage runner. Returns the output path."""
    if name == "ingest":
        from experiments.cross_sectional_part_marker.src.ingest import run_ingest
        return run_ingest(config, force=force)

    elif name == "rubric":
        from experiments.cross_sectional_part_marker.src.rubric_compiler import run_rubric_compiler
        return run_rubric_compiler(config, force=force)

    elif name == "analyse":
        from experiments.cross_sectional_part_marker.src.structural_analysis import run_structural_analysis
        return run_structural_analysis(config, force=force)

    elif name == "embed":
        from experiments.cross_sectional_part_marker.src.embeddings import run_embeddings
        return run_embeddings(config, force=force)

    elif name == "cluster":
        from experiments.cross_sectional_part_marker.src.clustering import run_clustering
        return run_clustering(config, force=force)

    elif name == "score":
        from experiments.cross_sectional_part_marker.src.scoring import run_scoring
        return run_scoring(config, force=force)

    elif name == "calibrate":
        from experiments.cross_sectional_part_marker.src.calibration import run_calibration
        return run_calibration(config, force=force)

    elif name == "feedback":
        from experiments.cross_sectional_part_marker.src.feedback import run_feedback
        return run_feedback(config, force=force)

    elif name == "validate":
        from experiments.cross_sectional_part_marker.src.validation import run_validation
        return run_validation(config, force=force)

    elif name == "export":
        from experiments.cross_sectional_part_marker.src.export import run_export
        return run_export(config, force=force)

    else:
        raise ValueError(f"Unknown stage: {name!r}. Valid stages: {STAGE_ORDER}")


def run_pipeline(
    config: PipelineConfig,
    stages: list[str],
    force: bool = False,
) -> None:
    """
    Run the requested pipeline stages in order.

    Parameters
    ----------
    config: PipelineConfig
    stages: List of stage names to run (subset of STAGE_ORDER)
    force:  If True, re-run even if output already exists
    """
    # Validate stage names
    unknown = [s for s in stages if s not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid: {STAGE_ORDER}")

    # Run in canonical order regardless of input order
    ordered_stages = [s for s in STAGE_ORDER if s in stages]

    logger.info("Pipeline starting: %d stages to run", len(ordered_stages))
    logger.info("Stages: %s", ", ".join(ordered_stages))
    logger.info("Output folder: %s", config.output_folder)

    wall_start = time.time()
    results: dict[str, str] = {}

    for stage in ordered_stages:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("STAGE: %s", stage.upper())
        logger.info("=" * 60)
        try:
            output = _run_stage(stage, config, force)
            elapsed = time.time() - t0
            if isinstance(output, dict):
                for k, v in output.items():
                    results[k] = str(v)
                    logger.info("  output: %s -> %s", k, v)
            elif output is not None:
                results[stage] = str(output)
                logger.info("  output: %s", output)
            logger.info("STAGE %s complete in %.1fs", stage.upper(), elapsed)
        except FileNotFoundError as exc:
            logger.error("STAGE %s failed — prerequisite missing: %s", stage.upper(), exc)
            logger.error("Stopping pipeline.")
            sys.exit(1)
        except Exception as exc:
            logger.exception("STAGE %s failed with unexpected error: %s", stage.upper(), exc)
            logger.error("Stopping pipeline.")
            sys.exit(1)

    total = time.time() - wall_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", total)
    logger.info("=" * 60)
    for k, v in results.items():
        logger.info("  %s -> %s", k, v)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-sectional part marker pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True, help="Path to YAML pipeline config")
    p.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER),
        help=f"Comma-separated list of stages to run. Default: all. Available: {','.join(STAGE_ORDER)}",
    )
    p.add_argument("--force", action="store_true", help="Re-run stages even if output exists")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    config = PipelineConfig.model_validate(raw_cfg)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    run_pipeline(config, stages, force=args.force)
