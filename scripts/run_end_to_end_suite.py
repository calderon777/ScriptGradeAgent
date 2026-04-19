import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marking_pipeline.core import (  # noqa: E402
    DEFAULT_OLLAMA_URL,
    _run_part_analysis_with_retry,
    apply_part_verifier_result,
    build_local_final_result,
    build_missing_part_analysis,
    build_submission_diagnostics,
    clear_prepared_assessment_map_cache,
    describe_moderation_plan,
    describe_structure_detection_mode,
    moderate_linked_part_analyses,
    prepare_assessment_map,
    refine_part_task_types_with_model,
    verify_part_analysis,
)
from scripts.run_human_benchmark import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_SAMPLE_PATHS,
    DEFAULT_WORKBOOK,
    BenchmarkResult,
    DebugStageRecord,
    apply_variant_to_parts,
    build_aggregate,
    build_summary_rows,
    build_variant_comparison_rows,
    compare_part_breakdowns,
    feedback_excerpt,
    load_context,
    load_human_records,
    log_debug_stage,
    model_part_breakdown,
    participant_id_from_path,
)
from scripts.run_human_benchmark import (  # noqa: E402
    build_submission_texts_from_path,
    detect_submission_parts,
    refine_submission_granularity,
    segment_submission_parts,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "end_to_end_suite"
DEFAULT_SAMPLE_LIST = ROOT_DIR / "five_script_smoke_2026-04-18.txt"
LAST_INGEST_MANIFEST = ROOT_DIR / ".scriptgrade_cache" / "last_ingest" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-script end-to-end suite with per-script debug artifacts and M2/M3 issue summaries."
    )
    parser.add_argument("--workbook", default=str(DEFAULT_WORKBOOK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sample", action="append", default=[], help="Submission path. May be passed multiple times.")
    parser.add_argument("--sample-list", default="", help="Optional text file containing one submission path per line.")
    parser.add_argument("--assessment-root", default="", help="Optional root folder to recursively discover all supported submission files.")
    parser.add_argument(
        "--use-manifest-assessment-root",
        action="store_true",
        help="Load the assessment root from .scriptgrade_cache/last_ingest/manifest.json and run the full corpus.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=("cached_map_only", "prepared_artifact_enriched"),
        default=[],
        help="Benchmark variant to run. May be passed multiple times. Defaults to both.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--task-type-model", default=None)
    parser.add_argument("--part-verifier-model", default=None)
    return parser.parse_args()


def resolve_sample_paths(args: argparse.Namespace) -> list[Path]:
    if args.sample:
        return [Path(item) for item in args.sample]
    if args.sample_list:
        sample_list = Path(args.sample_list)
        return [Path(line.strip()) for line in sample_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.assessment_root:
        root = Path(args.assessment_root)
        return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"})
    if args.use_manifest_assessment_root and LAST_INGEST_MANIFEST.exists():
        manifest = json.loads(LAST_INGEST_MANIFEST.read_text(encoding="utf-8"))
        root = Path(str(manifest.get("assessment_root", "")).strip())
        if root.exists():
            return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".txt"})
    if DEFAULT_SAMPLE_LIST.exists():
        return [Path(line.strip()) for line in DEFAULT_SAMPLE_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(DEFAULT_SAMPLE_PATHS)


def resolve_variants(args: argparse.Namespace) -> list[str]:
    return args.variant or ["cached_map_only", "prepared_artifact_enriched"]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "sample"


def part_score_for_row(part: Any, analysis: dict[str, Any]) -> float:
    score = analysis.get("provisional_score")
    if score is None and analysis.get("provisional_score_0_to_100") is not None and part.max_mark is not None:
        score = (float(analysis["provisional_score_0_to_100"]) * float(part.max_mark)) / 100.0
    return round(float(score or 0.0), 2)


def normalize_error_signature(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        return ""
    text = re.sub(r"Part\s+\d+(?:\s+Q\d+)?", "Part <section>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:240]


def summarize_debug_payload(payload: dict[str, Any], parts: list[Any], variant: str, run_index: int) -> BenchmarkResult:
    model_total = float(payload["model_total"])
    human_total = float(payload["human_total"])
    model_breakdown = {
        row["label"]: round(float(row["score"]), 2)
        for row in payload["part_debug_rows"]
        if row.get("score") is not None
    }
    return BenchmarkResult(
        variant=variant,
        run_index=run_index,
        path=str(payload["path"]),
        participant_id=str(payload["participant_id"]),
        full_name=str(payload["full_name"]),
        file_type=Path(str(payload["path"])).suffix.lower(),
        human_total=human_total,
        model_total=model_total,
        total_delta=round(model_total - human_total, 2),
        abs_total_delta=round(abs(model_total - human_total), 2),
        human_part_breakdown={},
        model_part_breakdown=model_breakdown,
        part_abs_delta_sum=float(payload["part_abs_delta_sum"]),
        part_abs_delta_mean=float(payload["part_abs_delta_mean"]),
        detected_part_labels=[str(item) for item in payload["detected_part_labels"]],
        scoring_word_count=sum(len(part.section_text.split()) for part in parts),
        structure_word_count=int(payload.get("structure_word_count", 0) or 0),
        preparation_seconds=float(payload["preparation_seconds"]),
        extraction_seconds=float(payload["extraction_seconds"]),
        latency_total=float(payload["scoring_total_seconds"]),
        latency_wall_total=float(payload["total_wall_seconds"]),
        latency_structure_detect=float(payload["structure_detect_seconds"]),
        latency_part_refine=float(payload["refine_seconds"]),
        latency_part_analysis=round(
            sum(
                float(row["seconds"])
                for row in payload["part_debug_rows"]
                if row.get("seconds") is not None
            ),
            2,
        ),
        latency_moderation=float(payload["moderation_seconds"]),
        latency_finalize=float(payload["finalize_seconds"]),
        validation_notes=[str(item) for item in payload["validation_notes"]],
        human_feedback_excerpt="",
        model_feedback_excerpt=str(payload.get("overall_feedback_excerpt", "")),
    )


def run_single_submission_suite(
    path: Path,
    context: Any,
    prepared_assessment_map: Any,
    preparation_seconds: float,
    human_record: Any,
    variant: str,
    model_name: str,
    task_type_model: str | None,
    part_verifier_model: str | None,
) -> tuple[dict[str, Any], list[Any]]:
    debug_records: list[DebugStageRecord] = []

    started = time.perf_counter()
    script_text, structure_text = build_submission_texts_from_path(path)
    extraction_seconds = log_debug_stage(
        debug_records,
        "extract_submission_text",
        started,
        f"scoring_words={len(script_text.split())}, structure_words={len((structure_text or script_text).split())}",
    )
    structure_input_text = structure_text or script_text

    scoring_started = time.perf_counter()
    structure_detection_mode = describe_structure_detection_mode(structure_input_text, context)

    started = time.perf_counter()
    detected_parts = detect_submission_parts(
        structure_input_text,
        path.name,
        model_name,
        context=context,
        ollama_url=DEFAULT_OLLAMA_URL,
    )
    structure_detect_seconds = log_debug_stage(
        debug_records,
        "detect_submission_parts",
        started,
        f"mode={structure_detection_mode['mode']}, detected_parts={len(detected_parts)}",
    )

    started = time.perf_counter()
    parts = segment_submission_parts(script_text, detected_parts)
    segment_seconds = log_debug_stage(
        debug_records,
        "segment_submission_parts",
        started,
        f"segmented_parts={len(parts)}",
    )

    started = time.perf_counter()
    parts = refine_submission_granularity(parts, context, path.name, model_name, DEFAULT_OLLAMA_URL)
    if task_type_model:
        parts = refine_part_task_types_with_model(parts, task_type_model, ollama_url=DEFAULT_OLLAMA_URL)
    refine_seconds = log_debug_stage(
        debug_records,
        "refine_submission_granularity",
        started,
        f"refined_parts={len(parts)}",
    )

    started = time.perf_counter()
    parts = apply_variant_to_parts(parts, prepared_assessment_map, variant)
    apply_seconds = log_debug_stage(
        debug_records,
        "apply_assessment_artifact",
        started,
        f"variant={variant}",
    )

    started = time.perf_counter()
    diagnostics = build_submission_diagnostics(script_text, parts)
    diagnostics_seconds = log_debug_stage(
        debug_records,
        "build_submission_diagnostics",
        started,
        f"detected_part_count={diagnostics.detected_part_count}",
    )

    part_analyses: list[dict[str, Any]] = []
    part_debug_rows: list[dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        started = time.perf_counter()
        analysis_error = ""
        if not part.section_text.strip():
            analysis = build_missing_part_analysis(part)
            seconds = log_debug_stage(
                debug_records,
                "part_analysis_missing",
                started,
                f"{index}/{len(parts)} {part.label}",
            )
        else:
            try:
                analysis = _run_part_analysis_with_retry(
                    part=part,
                    context=context,
                    filename=path.name,
                    model_name=model_name,
                    ollama_url=DEFAULT_OLLAMA_URL,
                )
                if part_verifier_model:
                    verifier_result = verify_part_analysis(
                        part=part,
                        context=context,
                        filename=path.name,
                        part_analysis=analysis,
                        model_name=part_verifier_model,
                        ollama_url=DEFAULT_OLLAMA_URL,
                    )
                    analysis = apply_part_verifier_result(analysis, verifier_result)
                seconds = log_debug_stage(
                    debug_records,
                    "part_analysis_model",
                    started,
                    f"{index}/{len(parts)} {part.label}",
                )
            except Exception as exc:
                analysis_error = str(exc)
                analysis = build_missing_part_analysis(part)
                analysis["coverage_comment"] = (
                    f"Model analysis failed in suite run ({type(exc).__name__}); "
                    "recorded as missing for artifact continuity."
                )
                seconds = log_debug_stage(
                    debug_records,
                    "part_analysis_error",
                    started,
                    f"{index}/{len(parts)} {part.label}: {type(exc).__name__}",
                )
        part_analyses.append(analysis)
        part_debug_rows.append(
            {
                "index": index,
                "label": part.label,
                "seconds": seconds,
                "word_count": len(part.section_text.split()),
                "score": analysis.get("provisional_score"),
                "score_pct": analysis.get("provisional_score_0_to_100"),
                "error": analysis_error,
                "error_signature": normalize_error_signature(analysis_error),
            }
        )

    moderation_debug_rows: list[dict[str, Any]] = []
    moderation_error = ""
    started = time.perf_counter()
    if len(part_analyses) > 1:
        moderation_plan = describe_moderation_plan(parts, part_analyses, prepared_assessment_map.prepared_map)
        before_scores = model_part_breakdown(parts, part_analyses)
        grouped_started = time.perf_counter()
        try:
            moderated = moderate_linked_part_analyses(
                script_text=script_text,
                context=context,
                filename=path.name,
                parts=parts,
                part_analyses=part_analyses,
                assessment_map=prepared_assessment_map.prepared_map,
                model_name=model_name,
                ollama_url=DEFAULT_OLLAMA_URL,
            )
            moderation_seconds = log_debug_stage(
                debug_records,
                "moderate_linked_part_analyses",
                grouped_started,
                "; ".join(f"{item['dependency_group']}={item['reason']}" for item in moderation_plan) or "no_linked_groups",
            )
        except Exception as exc:
            moderated = part_analyses
            moderation_error = f"{type(exc).__name__}: {exc}"
            moderation_seconds = log_debug_stage(
                debug_records,
                "moderation_error",
                grouped_started,
                f"{type(exc).__name__}",
            )
        after_scores = model_part_breakdown(parts, moderated)
        for item in moderation_plan:
            labels = [str(label) for label in item["part_labels"]]
            delta = round(sum(after_scores.get(label, 0.0) - before_scores.get(label, 0.0) for label in labels), 2)
            changed_labels = [label for label in labels if round(after_scores.get(label, 0.0) - before_scores.get(label, 0.0), 2) != 0.0]
            moderation_debug_rows.append(
                {
                    "seconds": moderation_seconds,
                    "dependency_group": item["dependency_group"],
                    "part_labels": labels,
                    "should_moderate": bool(item["should_moderate"]),
                    "reason": item["reason"],
                    "pre_group_score": round(sum(before_scores.get(label, 0.0) for label in labels), 2),
                    "post_group_score": round(sum(after_scores.get(label, 0.0) for label in labels), 2),
                    "group_score_delta": delta,
                    "changed_label_count": len(changed_labels),
                    "changed_labels": changed_labels,
                    "error": moderation_error,
                }
            )
        part_analyses = moderated
    else:
        moderation_seconds = log_debug_stage(debug_records, "moderate_linked_part_analyses", started, "skipped")

    started = time.perf_counter()
    result = build_local_final_result(
        context=context,
        diagnostics=diagnostics,
        parts=parts,
        part_analyses=part_analyses,
    )
    finalize_seconds = log_debug_stage(debug_records, "build_local_final_result", started)

    scoring_total_seconds = round(time.perf_counter() - scoring_started, 2)
    total_wall_seconds = round(extraction_seconds + preparation_seconds + scoring_total_seconds, 2)
    model_breakdown = model_part_breakdown(parts, part_analyses)
    part_abs_delta_sum, part_abs_delta_mean = compare_part_breakdowns(human_record.part_breakdown, model_breakdown)
    payload = {
        "path": str(path),
        "participant_id": human_record.participant_id,
        "full_name": human_record.full_name,
        "variant": variant,
        "preparation_seconds": preparation_seconds,
        "extraction_seconds": extraction_seconds,
        "structure_detect_seconds": structure_detect_seconds,
        "structure_detection_mode": structure_detection_mode,
        "segment_seconds": segment_seconds,
        "refine_seconds": refine_seconds,
        "apply_artifact_seconds": apply_seconds,
        "diagnostics_seconds": diagnostics_seconds,
        "moderation_seconds": moderation_seconds,
        "moderation_error": moderation_error,
        "finalize_seconds": finalize_seconds,
        "scoring_total_seconds": scoring_total_seconds,
        "total_wall_seconds": total_wall_seconds,
        "scoring_word_count": len(script_text.split()),
        "structure_word_count": len(structure_input_text.split()),
        "detected_part_labels": [part.label for part in parts],
        "human_total": human_record.total_mark,
        "model_total": result["total_mark"],
        "abs_total_delta": round(abs(float(result["total_mark"]) - float(human_record.total_mark)), 2),
        "part_abs_delta_sum": part_abs_delta_sum,
        "part_abs_delta_mean": part_abs_delta_mean,
        "validation_notes": list(result.get("validation_notes", [])),
        "overall_feedback_excerpt": feedback_excerpt(str(result.get("overall_feedback", ""))),
        "part_debug_rows": part_debug_rows,
        "moderation_debug_rows": moderation_debug_rows,
        "debug_stages": [asdict(record) for record in debug_records],
    }
    return payload, parts


def write_payload_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "debug_run.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    pd.DataFrame(payload["part_debug_rows"]).to_csv(output_dir / "debug_parts.csv", index=False)
    pd.DataFrame(payload["debug_stages"]).to_csv(output_dir / "debug_stages.csv", index=False)
    pd.DataFrame(payload["moderation_debug_rows"]).to_csv(output_dir / "moderation_groups.csv", index=False)


def main() -> None:
    args = parse_args()
    workbook_path = Path(args.workbook)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = resolve_sample_paths(args)
    variants = resolve_variants(args)
    human_records = load_human_records(workbook_path)
    context = load_context()

    results: list[BenchmarkResult] = []
    part_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    moderation_rows: list[dict[str, Any]] = []
    failed_submissions: list[dict[str, Any]] = []

    for variant in variants:
        clear_prepared_assessment_map_cache()
        prep_started = time.perf_counter()
        prepared_assessment_map = prepare_assessment_map(
            context=context,
            verifier_model_name=args.verifier_model,
        )
        preparation_seconds = round(time.perf_counter() - prep_started, 2)
        for run_index in range(1, args.repeats + 1):
            for index, path in enumerate(sample_paths, start=1):
                participant_id = participant_id_from_path(path)
                human_record = human_records[participant_id]
                print(f"[suite {variant} run {run_index} {index}/{len(sample_paths)}] {path.name} :: {human_record.full_name}", flush=True)
                try:
                    payload, parts = run_single_submission_suite(
                        path=path,
                        context=context,
                        prepared_assessment_map=prepared_assessment_map,
                        preparation_seconds=preparation_seconds,
                        human_record=human_record,
                        variant=variant,
                        model_name=args.model,
                        task_type_model=args.task_type_model,
                        part_verifier_model=args.part_verifier_model,
                    )
                except Exception as exc:
                    failed_submissions.append(
                        {
                            "variant": variant,
                            "run_index": run_index,
                            "participant_id": participant_id,
                            "path": str(path),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    print(f"[suite-error] {path.name} :: {type(exc).__name__}: {exc}", flush=True)
                    continue
                sample_slug = slugify(f"{participant_id}_{path.stem}")
                sample_output_dir = output_dir / variant / f"run_{run_index}" / sample_slug
                write_payload_artifacts(sample_output_dir, payload)
                results.append(summarize_debug_payload(payload, parts, variant, run_index))

                for row in payload["part_debug_rows"]:
                    part_rows.append(
                        {
                            "variant": variant,
                            "run_index": run_index,
                            "participant_id": participant_id,
                            "path": str(path),
                            **row,
                        }
                    )
                for row in payload["debug_stages"]:
                    stage_rows.append(
                        {
                            "variant": variant,
                            "run_index": run_index,
                            "participant_id": participant_id,
                            "path": str(path),
                            **row,
                        }
                    )
                for row in payload["moderation_debug_rows"]:
                    moderation_rows.append(
                        {
                            "variant": variant,
                            "run_index": run_index,
                            "participant_id": participant_id,
                            "path": str(path),
                            **row,
                        }
                    )

    summary_rows = build_summary_rows(results)
    comparison_rows = build_variant_comparison_rows(results)
    result_rows = [asdict(item) for item in results]

    error_rows = [row for row in part_rows if str(row.get("error", "")).strip()]
    error_signature_counts: dict[str, int] = {}
    for row in error_rows:
        signature = str(row.get("error_signature", "")).strip()
        if signature:
            error_signature_counts[signature] = error_signature_counts.get(signature, 0) + 1

    issue_summary = {
        "m2_scoring": {
            "scripts_with_any_part_error": len({(row["variant"], row["run_index"], row["participant_id"]) for row in error_rows}),
            "part_errors_total": len(error_rows),
            "top_error_signatures": [
                {"error_signature": key, "count": value}
                for key, value in sorted(error_signature_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
            ],
            "worst_total_delta_runs": sorted(
                (
                    {
                        "variant": row["variant"],
                        "run_index": row["run_index"],
                        "participant_id": row["participant_id"],
                        "path": row["path"],
                        "abs_total_delta": row["abs_total_delta"],
                        "model_total": row["model_total"],
                        "human_total": row["human_total"],
                    }
                    for row in result_rows
                ),
                key=lambda item: (-float(item["abs_total_delta"]), item["participant_id"]),
            )[:10],
        },
        "m3_latency": {
            "mean_scoring_total_seconds": round(
                sum(float(row["latency_total"]) for row in result_rows) / max(len(result_rows), 1),
                2,
            ) if result_rows else 0.0,
            "mean_moderation_seconds": round(
                sum(float(row["latency_moderation"]) for row in result_rows) / max(len(result_rows), 1),
                2,
            ) if result_rows else 0.0,
            "moderation_groups_total": len(moderation_rows),
            "moderation_groups_eligible": sum(1 for row in moderation_rows if row.get("should_moderate")),
            "moderation_groups_changed": sum(1 for row in moderation_rows if float(row.get("group_score_delta", 0.0)) != 0.0),
            "moderation_groups_zero_delta": sum(1 for row in moderation_rows if float(row.get("group_score_delta", 0.0)) == 0.0),
            "top_moderation_reasons": [
                {"reason": key, "count": value}
                for key, value in sorted(
                    {
                        str(row.get("reason", "")).strip(): sum(
                            1 for item in moderation_rows if str(item.get("reason", "")).strip() == str(row.get("reason", "")).strip()
                        )
                        for row in moderation_rows
                        if str(row.get("reason", "")).strip()
                    }.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:20]
            ],
        },
    }

    aggregate = {
        variant: build_aggregate([item for item in results if item.variant == variant])
        for variant in variants
    }
    suite_payload = {
        "workbook": str(workbook_path),
        "model_name": args.model,
        "verifier_model_name": args.verifier_model,
        "task_type_model_name": args.task_type_model,
        "part_verifier_model_name": args.part_verifier_model,
        "variants": variants,
        "repeats": args.repeats,
        "sample_paths": [str(path) for path in sample_paths],
        "aggregate": aggregate,
        "summary_rows": summary_rows,
        "comparison_rows": comparison_rows,
        "issue_summary": issue_summary,
        "failed_submissions": failed_submissions,
        "results": result_rows,
    }

    (output_dir / "summary.json").write_text(json.dumps(suite_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (output_dir / "sample_paths.txt").write_text("\n".join(str(path) for path in sample_paths), encoding="utf-8")
    pd.DataFrame(result_rows).to_csv(output_dir / "runs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "variant_comparison.csv", index=False)
    pd.DataFrame(part_rows).to_csv(output_dir / "part_debug_rows.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output_dir / "stage_debug_rows.csv", index=False)
    pd.DataFrame(moderation_rows).to_csv(output_dir / "moderation_debug_rows.csv", index=False)
    pd.DataFrame(error_rows).to_csv(output_dir / "part_errors.csv", index=False)
    pd.DataFrame(failed_submissions).to_csv(output_dir / "failed_submissions.csv", index=False)
    (output_dir / "issue_summary.json").write_text(json.dumps(issue_summary, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
