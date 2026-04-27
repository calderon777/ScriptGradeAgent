# Cross-Sectional Part Marker

## What this experiment does

This module implements a full rubric-first, cross-sectional marking pipeline for open-ended exam questions. Given a folder of student submissions (PDF or plain text) and a marking scheme, the pipeline automatically splits each submission into per-question-part answers, compiles a structured rubric, analyses each answer for required concepts and evidence, scores each answer against criteria using a local LLM, calibrates scores using cross-sectional embedding-based clustering and pairwise comparisons, and generates professional written feedback. All model inference runs locally via Ollama — no data is sent to any external API. A Streamlit review application allows a human marker to inspect, edit, and approve AI marks before export.

---

## Architecture: 10-Stage Pipeline

1. **Ingest** (`ingest.py`) — Extract text from PDF/.txt submissions; split into per-question-part `AnswerPart` records.
2. **Rubric Compiler** (`rubric_compiler.py`) — Parse marking scheme / rubric / instructions into a structured `RubricSpec`. Uses LLM extraction for free-text sources.
3. **Structural Analysis** (`structural_analysis.py`) — For each answer part, use a local LLM to identify concepts present/missing, misconceptions, evidence, and quality flags.
4. **Embeddings** (`embeddings.py`) — Embed each cleaned answer and the rubric criteria; compute cosine similarities and find nearest neighbours.
5. **Clustering** (`clustering.py`) — Cluster answers per question part using HDBSCAN (or KMeans fallback); identify outliers and borderline cases; optionally project to 2D with UMAP.
6. **Scoring** (`scoring.py`) — Multi-step LLM scoring against each criterion; enforce score ≤ max per criterion; assign grade bucket.
7. **Calibration** (`calibration.py`) — Pairwise LLM comparison for borderline/discrepant NN pairs; update scores only when both structural and pairwise evidence agree.
8. **Feedback** (`feedback.py`) — Generate per-criterion written feedback: strengths, limitations, improvement advice, summary.
9. **Validation** (`validation.py`) — If human marks are provided, compute MAE, Pearson r, Spearman ρ, QWK, bucket agreement, and nearest-neighbour consistency warnings.
10. **Export** (`export.py`) — Produce `final_marks.csv`, `detailed_feedback.xlsx`, `moderation_report.html`, and `audit_trail.jsonl`.

---

## Running Each Stage

All stages can be run individually or together via `run_pipeline.py`.

### Full pipeline
```bash
python -m experiments.cross_sectional_part_marker.src.run_pipeline \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

# Run specific stages only
python -m experiments.cross_sectional_part_marker.src.run_pipeline \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml \
    --stages ingest,rubric,analyse,score,calibrate,feedback

# Force re-run even if outputs exist
python -m experiments.cross_sectional_part_marker.src.run_pipeline \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml \
    --force
```

### Individual stages
```bash
python -m experiments.cross_sectional_part_marker.src.ingest \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.rubric_compiler \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.structural_analysis \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.embeddings \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.clustering \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.scoring \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.calibration \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.feedback \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml

python -m experiments.cross_sectional_part_marker.src.validation \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml \
    --human-marks path/to/human_marks.csv   # optional

python -m experiments.cross_sectional_part_marker.src.export \
    --config experiments/cross_sectional_part_marker/config/experiment_cross_sectional_marker.yaml
```

All individual stage CLIs also accept `--force` and `--log-level`.

---

## Running the Review App

```bash
streamlit run experiments/cross_sectional_part_marker/app/review_app.py
```

The app requires the pipeline to have been run at least through the calibration stage. It provides:
- **Cohort Overview**: score/bucket distributions, cluster summaries, flagged-for-review list
- **Review Answers**: view each answer with rubric, AI scores, feedback, and nearest neighbours; edit scores/feedback; approve or flag
- **Export**: trigger the export stage and download `final_marks.csv` and `detailed_feedback.xlsx`

---

## Config Reference

Key fields in `config/experiment_cross_sectional_marker.yaml`:

| Field | Description |
|---|---|
| `assessment_name` | Human-readable assessment name |
| `submissions_folder` | Path to folder containing PDF/.txt submissions |
| `output_folder` | Path where all stage outputs are written |
| `marking_scheme_path` | Path to marking scheme (JSON/YAML for structured, or plain text for LLM extraction) |
| `rubric_path` | (Optional) Additional rubric document |
| `instructions_path` | (Optional) Additional instructions document |
| `sample_answers_path` | (Optional) JSONL of sample/anchor answers for embedding comparison |
| `question_splitting.method` | `heuristic`, `llm`, or `hybrid` |
| `question_splitting.patterns` | Regex patterns to identify question headings |
| `question_splitting.llm_fallback_threshold` | Confidence threshold below which hybrid mode uses LLM |
| `models.analysis_model` | Ollama model for structural analysis |
| `models.scoring_model` | Ollama model for scoring and calibration |
| `models.feedback_model` | Ollama model for feedback generation |
| `models.embedding_model` | Ollama model for embeddings |
| `ollama.base_url` | Ollama server URL (default: `http://localhost:11434`) |
| `ollama.timeout` | Request timeout in seconds |
| `temperature` | LLM sampling temperature (0.0–2.0) |
| `confidence_threshold` | Below this confidence, answers are flagged for human review |
| `run_clustering` | Enable cross-sectional clustering |
| `run_umap` | Enable UMAP 2D projection (requires `umap-learn`) |
| `run_pairwise_boundary` | Enable pairwise LLM calibration for borderline cases |

---

## Methodological Position

This pipeline adopts a **rubric-first, cross-sectional** approach to automated marking:

- **Rubric-first**: Every scoring decision is grounded in explicit rubric criteria with required evidence, partial credit rules, and no-credit conditions. The LLM is never asked to "give a mark" without a rubric; it is asked to evaluate specific criteria and produce structured evidence before assigning a score.

- **Cross-sectional**: Embeddings and clustering are used to understand the population of answers before scoring — not to determine scores. The clustering stage identifies answer quality bands, outliers, and borderline cases to focus human review. Scores themselves come from rubric criteria evaluation; similarity scores are used for calibration moderation only.

- **Calibration by disagreement**: Pairwise comparisons are triggered only when nearest-neighbour answers receive discrepant scores (> 1 mark). Score adjustments are made only when both structural analysis and a pairwise LLM comparison agree on direction, with confidence ≥ 0.7. Every change (and every non-change) is documented.

- **Human review by exception**: The pipeline flags answers for human review based on low confidence, borderline cluster membership, or NN score discrepancy. The review app makes it easy to act on these flags without reviewing every answer.

---

## Privacy

All model calls are made locally via Ollama (`http://localhost:11434`). No student answer text, personal data, or marking scheme content is transmitted to any external server or third-party API. The `OllamaClient` has no capability to call external endpoints and will raise `OllamaConnectionError` if Ollama is not running locally.
