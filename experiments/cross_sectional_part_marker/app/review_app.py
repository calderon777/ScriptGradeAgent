"""
Streamlit human-review application for the cross-sectional part marker.

Run with:
    streamlit run experiments/cross_sectional_part_marker/app/review_app.py

The app expects pipeline outputs to be present in the output folder
defined in the config YAML. If outputs are missing it shows a clear
"No data found — run pipeline first" message.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: ensure project root is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent  # ScriptGradeAgent/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import yaml

from experiments.cross_sectional_part_marker.src.schemas import (
    AnswerPart,
    CalibratedAnswer,
    CrossSectionalStructure,
    EmbeddingFeatures,
    Feedback,
    FeedbackContent,
    HumanReviewEntry,
    PipelineConfig,
    RubricSpec,
    ScoredAnswer,
)
from experiments.cross_sectional_part_marker.src.embeddings import answer_key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = _HERE.parent / "config" / "experiment_cross_sectional_marker.yaml"
_BUCKETS = ["A", "B", "C", "D", "E", "F"]


# ---------------------------------------------------------------------------
# Data loading helpers (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_data(show_spinner=False)
def _load_jsonl_raw(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


@st.cache_data(show_spinner=False)
def _load_rubric(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _load_structure(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _invalidate_caches() -> None:
    """Clear all cached data after a save."""
    _load_jsonl_raw.clear()
    _load_rubric.clear()
    _load_structure.clear()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Cross-Sectional Part Marker — Review", layout="wide")
    st.title("Cross-Sectional Part Marker — Human Review")

    # ------------------------------------------------------------------
    # Sidebar: config selection
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        config_path_input = st.text_input(
            "Config YAML path",
            value=str(_DEFAULT_CONFIG),
        )

        config_path = Path(config_path_input)
        if not config_path.exists():
            st.error(f"Config not found: {config_path}")
            st.stop()

        raw_cfg = _load_config(str(config_path))
        try:
            cfg = PipelineConfig.model_validate(raw_cfg)
        except Exception as exc:
            st.error(f"Invalid config: {exc}")
            st.stop()

        output_dir = Path(cfg.output_folder)
        st.caption(f"Output folder: `{output_dir}`")

        # Question / part filter
        calibrated_raw = _load_jsonl_raw(str(output_dir / "calibrated_scores.jsonl"))
        if not calibrated_raw:
            st.warning("No calibrated scores found. Run the pipeline first.")
            st.stop()

        all_qids = sorted({r["question_id"] for r in calibrated_raw})
        selected_qid = st.selectbox("Question ID", options=["(all)"] + all_qids)

        all_pids = sorted({r["part_id"] for r in calibrated_raw})
        selected_pid = st.selectbox("Part ID", options=["(all)"] + all_pids)

        reviewer_name = st.text_input("Reviewer name", value="reviewer1")

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    calibrated_list = [CalibratedAnswer.model_validate(r) for r in calibrated_raw]
    feedback_raw = _load_jsonl_raw(str(output_dir / "feedback.jsonl"))
    feedback_list = [Feedback.model_validate(r) for r in feedback_raw]
    parts_raw = _load_jsonl_raw(str(output_dir / "answer_parts.jsonl"))
    parts_list = [AnswerPart.model_validate(r) for r in parts_raw]
    embedding_raw = _load_jsonl_raw(str(output_dir / "embedding_features.jsonl"))
    embedding_list = [EmbeddingFeatures.model_validate(r) for r in embedding_raw]
    scores_raw = _load_jsonl_raw(str(output_dir / "criterion_scores.jsonl"))
    scores_list = [ScoredAnswer.model_validate(r) for r in scores_raw]
    structure_raw = _load_structure(str(output_dir / "cross_sectional_structure.json"))
    structures = [CrossSectionalStructure.model_validate(d) for d in structure_raw]
    rubric_raw = _load_rubric(str(output_dir / "rubric_spec.json"))
    rubric = RubricSpec.model_validate(rubric_raw) if rubric_raw else None

    # Build lookup maps
    feedback_map: dict[str, Feedback] = {
        answer_key(f.submission_id, f.question_id, f.part_id): f for f in feedback_list
    }
    parts_map: dict[str, AnswerPart] = {
        answer_key(p.submission_id, p.question_id, p.part_id): p for p in parts_list
    }
    emb_map: dict[str, EmbeddingFeatures] = {
        answer_key(e.submission_id, e.question_id, e.part_id): e for e in embedding_list
    }
    score_map: dict[str, ScoredAnswer] = {
        answer_key(s.submission_id, s.question_id, s.part_id): s for s in scores_list
    }

    # Filter calibrated answers
    filtered = [
        c for c in calibrated_list
        if (selected_qid == "(all)" or c.question_id == selected_qid)
        and (selected_pid == "(all)" or c.part_id == selected_pid)
    ]

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tab_overview, tab_review, tab_export = st.tabs(["Cohort Overview", "Review Answers", "Export"])

    # ================================================================
    # TAB 1: Cohort Overview
    # ================================================================
    with tab_overview:
        st.header("Cohort Overview")

        if not filtered:
            st.info("No answers match current filter.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Answers", len(filtered))
            col2.metric("Flagged for Review", sum(1 for c in filtered if c.needs_human_review))
            scores = [c.calibrated_score for c in filtered]
            col3.metric("Mean Score", f"{sum(scores)/len(scores):.2f}" if scores else "—")
            col4.metric("Score Range", f"{min(scores):.1f} – {max(scores):.1f}" if scores else "—")

            st.subheader("Score Distribution")
            try:
                import altair as alt
                import pandas as pd

                df_scores = pd.DataFrame({"score": scores})
                chart = (
                    alt.Chart(df_scores)
                    .mark_bar()
                    .encode(
                        alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Score"),
                        alt.Y("count():Q", title="Count"),
                    )
                    .properties(width=500, height=250)
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                from collections import Counter
                import math
                st.caption("(altair not installed — showing table)")
                rounded = [round(s) for s in scores]
                freq = Counter(rounded)
                st.bar_chart(freq)

            st.subheader("Bucket Distribution")
            from collections import Counter
            bucket_freq = Counter(c.calibrated_bucket for c in filtered)
            bucket_data = {b: bucket_freq.get(b, 0) for b in _BUCKETS}
            st.bar_chart(bucket_data)

            st.subheader("Cluster Summaries")
            if not structures:
                st.info("No clustering data found. Run the clustering stage first.")
            else:
                for struct in structures:
                    if (selected_qid != "(all)" and struct.question_id != selected_qid):
                        continue
                    if (selected_pid != "(all)" and struct.part_id != selected_pid):
                        continue
                    st.markdown(f"**{struct.question_id}{struct.part_id}** — {struct.n_answers} answers, "
                                f"{len(struct.outliers)} outliers, {len(struct.borderline_cases)} borderline")
                    if struct.clusters:
                        import pandas as pd
                        cluster_df = pd.DataFrame([
                            {
                                "Cluster": cl.cluster_id,
                                "Size": cl.size,
                                "Quality Band": cl.likely_quality_band,
                                "Summary": cl.summary,
                            }
                            for cl in struct.clusters
                        ])
                        st.dataframe(cluster_df, use_container_width=True)

            st.subheader("Flagged for Review")
            flagged = [c for c in filtered if c.needs_human_review]
            if flagged:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame([
                        {
                            "submission_id": c.submission_id,
                            "question": f"{c.question_id}{c.part_id}",
                            "score": c.calibrated_score,
                            "bucket": c.calibrated_bucket,
                            "notes": c.calibration_notes[:60],
                        }
                        for c in flagged
                    ]),
                    use_container_width=True,
                )
            else:
                st.success("No answers flagged for review in current selection.")

    # ================================================================
    # TAB 2: Review Answers
    # ================================================================
    with tab_review:
        st.header("Review Individual Answers")

        if not filtered:
            st.info("No answers match current filter.")
        else:
            submission_options = [f"{c.submission_id} — {c.question_id}{c.part_id}" for c in filtered]
            selected_idx = st.selectbox("Select submission", options=range(len(submission_options)),
                                        format_func=lambda i: submission_options[i])
            cal = filtered[selected_idx]
            key = answer_key(cal.submission_id, cal.question_id, cal.part_id)

            part = parts_map.get(key)
            fb = feedback_map.get(key)
            emb = emb_map.get(key)
            scored = score_map.get(key)
            spec = rubric.get_part_spec(cal.question_id, cal.part_id) if rubric else None

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.subheader("Student Answer")
                if part:
                    st.caption(f"Source: `{part.source_file}`")
                    st.markdown(f"**Word count:** {part.word_count} | "
                                f"**Extraction confidence:** {part.extraction_confidence:.2f}")
                    with st.expander("Raw text", expanded=True):
                        st.text(part.raw_text or "(empty)")
                    if part.extraction_notes:
                        st.warning(f"Extraction notes: {part.extraction_notes}")
                else:
                    st.warning("Answer part not found.")

                st.subheader("Rubric Criteria")
                if spec:
                    for c in spec.criteria:
                        st.markdown(
                            f"**{c.name}** (max {c.max_marks} marks): {c.description}"
                        )
                else:
                    st.info("No rubric criteria available for this part.")

                if emb and emb.nearest_neighbours:
                    st.subheader("Nearest Neighbours")
                    nn_rows = []
                    cal_by_sub: dict[str, float] = {
                        answer_key(c2.submission_id, c2.question_id, c2.part_id): c2.calibrated_score
                        for c2 in filtered
                    }
                    for nn_id in emb.nearest_neighbours:
                        nn_key = answer_key(nn_id, cal.question_id, cal.part_id)
                        nn_score = cal_by_sub.get(nn_key, "—")
                        nn_rows.append({"submission_id": nn_id, "score": nn_score})
                    import pandas as pd
                    st.dataframe(pd.DataFrame(nn_rows), use_container_width=True)

            with col_right:
                st.subheader("AI Scores")
                max_score = cal.max_score or (spec.max_marks if spec else 0.0)
                st.metric("AI Total Score", f"{cal.calibrated_score:.1f} / {max_score:.1f}")
                st.metric("Bucket", cal.calibrated_bucket)
                st.metric("Confidence", f"{cal.confidence:.2f}")
                if scored and scored.criterion_scores:
                    st.caption("Criterion-level proposed scores")
                    for cs in scored.criterion_scores:
                        st.markdown(
                            f"**{cs.criterion_id}:** {cs.score:g}/{cs.max_score:g} "
                            f"({cs.confidence:.2f})  \n{cs.reason}"
                        )
                if cal.calibration_notes:
                    st.info(f"Calibration notes: {cal.calibration_notes}")
                if cal.needs_human_review:
                    st.warning("Flagged for human review")

                if fb:
                    st.subheader("AI Feedback")
                    st.markdown(f"**Strengths:** {fb.feedback.strengths}")
                    st.markdown(f"**Limitations:** {fb.feedback.limitations}")
                    st.markdown(f"**Advice:** {fb.feedback.improvement_advice}")
                    st.markdown(f"**Summary:** {fb.feedback.summary}")

                st.divider()
                st.subheader("Human Review")

                # Session state key for this submission's edits
                ss_key = f"review_{key}"
                if ss_key not in st.session_state:
                    st.session_state[ss_key] = {
                        "human_score": cal.calibrated_score,
                        "human_bucket": cal.calibrated_bucket,
                        "strengths": fb.feedback.strengths if fb else "",
                        "limitations": fb.feedback.limitations if fb else "",
                        "advice": fb.feedback.improvement_advice if fb else "",
                        "review_notes": "",
                        "action": "pending",
                    }
                ed = st.session_state[ss_key]

                max_m = spec.max_marks if spec else 100.0
                ed["human_score"] = st.number_input(
                    "Human score", min_value=0.0, max_value=float(max_m),
                    value=float(ed["human_score"]), step=0.5,
                )
                ed["human_bucket"] = st.selectbox(
                    "Human bucket", options=_BUCKETS,
                    index=_BUCKETS.index(ed["human_bucket"]) if ed["human_bucket"] in _BUCKETS else 0,
                )

                if spec:
                    st.caption("Criterion scores (edit below)")
                    for c in spec.criteria:
                        current = 0.0
                        if scored:
                            match = next((cs for cs in scored.criterion_scores if cs.criterion_id == c.criterion_id), None)
                            if match:
                                current = match.score
                        ed[f"criterion_{c.criterion_id}"] = st.number_input(
                            f"{c.name} ({c.criterion_id})",
                            min_value=0.0,
                            max_value=float(c.max_marks),
                            value=float(ed.get(f"criterion_{c.criterion_id}", current)),
                            step=0.5,
                            key=f"criterion_{key}_{c.criterion_id}",
                        )

                ed["strengths"] = st.text_area("Strengths", value=ed["strengths"], height=80)
                ed["limitations"] = st.text_area("Limitations", value=ed["limitations"], height=80)
                ed["advice"] = st.text_area("Improvement advice", value=ed["advice"], height=80)
                ed["review_notes"] = st.text_area("Review notes", value=ed["review_notes"], height=60)

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("Approve as-is", key=f"approve_{key}"):
                        ed["action"] = "approved"
                        _save_review(ed, cal, fb, reviewer_name, output_dir, action="approved")
                        st.success("Approved.")
                    if st.button("Save edits", key=f"save_{key}"):
                        ed["action"] = "edited"
                        _save_review(ed, cal, fb, reviewer_name, output_dir, action="edited")
                        st.success("Edits saved.")
                    if st.button("Flag: needs second marker", key=f"flag_sm_{key}"):
                        ed["action"] = "flagged_second_marker"
                        _save_review(ed, cal, fb, reviewer_name, output_dir, action="flagged_second_marker")
                        st.warning("Flagged for second marker.")

                with col_b2:
                    if st.button("Flag: extraction error", key=f"flag_ex_{key}"):
                        ed["action"] = "flagged_extraction_error"
                        _save_review(ed, cal, fb, reviewer_name, output_dir, action="flagged_extraction_error")
                        st.warning("Flagged: extraction error.")
                    if st.button("Flag: rubric ambiguity", key=f"flag_rub_{key}"):
                        ed["action"] = "flagged_rubric_ambiguity"
                        _save_review(ed, cal, fb, reviewer_name, output_dir, action="flagged_rubric_ambiguity")
                        st.warning("Flagged: rubric ambiguity.")
                    if st.button("Flag: similarity concern", key=f"flag_sim_{key}"):
                        ed["action"] = "flagged_similarity_concern"
                        _save_review(ed, cal, fb, reviewer_name, output_dir, action="flagged_similarity_concern")
                        st.warning("Flagged: similarity concern.")

    # ================================================================
    # TAB 3: Export
    # ================================================================
    with tab_export:
        st.header("Export")

        if st.button("Run Export Stage"):
            try:
                from experiments.cross_sectional_part_marker.src.export import run_export
                with st.spinner("Exporting…"):
                    outputs = run_export(cfg, force=True)
                st.success("Export complete!")
                for name, path in outputs.items():
                    st.write(f"- **{name}**: `{path}`")
                _invalidate_caches()
            except Exception as exc:
                st.error(f"Export failed: {exc}")

        st.divider()

        csv_path = output_dir / "final_marks.csv"
        xlsx_path = output_dir / "detailed_feedback.xlsx"

        if csv_path.exists():
            with open(csv_path, "rb") as f:
                st.download_button("Download final_marks.csv", f, file_name="final_marks.csv", mime="text/csv")
        else:
            st.info("final_marks.csv not yet generated. Click 'Run Export Stage' above.")

        if xlsx_path.exists():
            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "Download detailed_feedback.xlsx", f,
                    file_name="detailed_feedback.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.info("detailed_feedback.xlsx not yet generated.")


def _save_review(
    edits: dict,
    cal: CalibratedAnswer,
    fb: Feedback | None,
    reviewer: str,
    output_dir: Path,
    action: str,
) -> None:
    """Append a HumanReviewEntry to human_review_log.jsonl."""
    criterion_scores = {
        key.replace("criterion_", "", 1): float(value)
        for key, value in edits.items()
        if key.startswith("criterion_")
    }
    human_feedback = "\n\n".join(
        part for part in [
            edits.get("strengths", ""),
            edits.get("limitations", ""),
            edits.get("advice", ""),
        ]
        if part
    )
    entry = HumanReviewEntry(
        review_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        reviewer=reviewer,
        submission_id=cal.submission_id,
        question_id=cal.question_id,
        part_id=cal.part_id,
        ai_score=cal.calibrated_score,
        human_score=edits.get("human_score"),
        human_criterion_scores=criterion_scores,
        ai_bucket=cal.calibrated_bucket,
        human_bucket=edits.get("human_bucket"),
        ai_feedback=fb.feedback.summary if fb else "",
        human_feedback=human_feedback,
        review_action=action,
        review_notes=edits.get("review_notes", ""),
        change_reason="",
    )
    log_path = output_dir / "human_review_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
