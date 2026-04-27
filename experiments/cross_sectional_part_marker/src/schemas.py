"""
Pydantic v2 data models for the cross-sectional part marker pipeline.

All data flowing through the pipeline is validated against these schemas.
Import order follows the data flow: AnswerPart -> Rubric -> Analysis -> Score -> Feedback.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Answer ingestion
# ---------------------------------------------------------------------------


class AnswerPart(BaseModel):
    """A single extracted question-part answer from a student submission."""

    submission_id: str = Field(..., description="Unique identifier for the submission file")
    anonymised_student_id: str = Field(..., description="Anonymised student identifier")
    source_file: str = Field(..., description="Path to the original submission file")
    question_id: str = Field(..., description="e.g. 'Q1'")
    part_id: str = Field(..., description="e.g. 'a', 'b', or '' for whole-question")
    raw_text: str = Field(..., description="Verbatim extracted text, never modified post-extraction")
    cleaned_text: str = Field(..., description="Lightly cleaned text used for analysis")
    word_count: int = Field(..., ge=0)
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_notes: str = Field(default="", description="Notes on extraction quality or split ambiguity")


# ---------------------------------------------------------------------------
# Rubric structures
# ---------------------------------------------------------------------------


class Criterion(BaseModel):
    """A single marking criterion within a question part."""

    criterion_id: str
    name: str
    description: str
    max_marks: float = Field(..., ge=0.0)
    required_evidence: list[str] = Field(default_factory=list)
    partial_credit_rules: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)
    no_credit_conditions: list[str] = Field(default_factory=list)
    provenance: str = Field(default="", description="Source text this criterion was derived from")


class BucketDefinition(BaseModel):
    """Defines one grade bucket (A–F) for a question part."""

    bucket: Literal["A", "B", "C", "D", "E", "F"]
    label: str = Field(..., description="Human-readable label e.g. 'Excellent'")
    mark_range: tuple[float, float] = Field(..., description="Inclusive [min, max] mark range")
    description: str = Field(..., description="Qualitative description of answers in this bucket")

    @field_validator("mark_range")
    @classmethod
    def validate_mark_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError(f"mark_range min ({v[0]}) must be <= max ({v[1]})")
        return v


class QuestionPartSpec(BaseModel):
    """Full specification for one question part including all criteria and buckets."""

    question_id: str
    part_id: str
    max_marks: float = Field(..., ge=0.0)
    task_instruction: str
    criteria: list[Criterion] = Field(default_factory=list)
    bucket_definitions: list[BucketDefinition] = Field(default_factory=list)


class RubricSpec(BaseModel):
    """Top-level rubric specification for an entire assessment."""

    assessment_name: str
    needs_human_confirmation: bool = Field(
        default=False,
        description="True if rubric was AI-generated from vague source and needs verification",
    )
    question_parts: list[QuestionPartSpec] = Field(default_factory=list)

    def get_part_spec(self, question_id: str, part_id: str) -> QuestionPartSpec | None:
        """Retrieve a specific QuestionPartSpec by question_id and part_id."""
        for qp in self.question_parts:
            if qp.question_id == question_id and qp.part_id == part_id:
                return qp
        return None


# ---------------------------------------------------------------------------
# Structural analysis
# ---------------------------------------------------------------------------


class AnswerAnalysis(BaseModel):
    """Structured analysis of what a student answer contains and lacks."""

    submission_id: str
    question_id: str
    part_id: str
    detected_claims: list[str] = Field(default_factory=list)
    required_concepts_present: list[str] = Field(default_factory=list)
    required_concepts_missing: list[str] = Field(default_factory=list)
    reasoning_steps_present: list[str] = Field(default_factory=list)
    reasoning_steps_missing: list[str] = Field(default_factory=list)
    evidence_or_examples: list[str] = Field(default_factory=list)
    calculation_or_method_steps: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)
    irrelevant_material: list[str] = Field(default_factory=list)
    unsupported_assertions: list[str] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    model_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Embeddings and clustering
# ---------------------------------------------------------------------------


class EmbeddingFeatures(BaseModel):
    """Embedding-derived features for a single answer part."""

    submission_id: str
    question_id: str
    part_id: str
    embedding_model: str
    nearest_neighbours: list[str] = Field(default_factory=list, description="submission_ids of top-5 neighbours")
    similarity_to_model_answer: float | None = None
    similarity_to_rubric: float | None = None
    similarity_to_anchor_A: float | None = None
    similarity_to_anchor_B: float | None = None
    similarity_to_anchor_C: float | None = None
    cluster_id: int | None = None
    outlier_score: float | None = None


class ClusterSummary(BaseModel):
    """Summary of a cluster of answers for a given question part."""

    cluster_id: int
    size: int = Field(..., ge=0)
    representative_submission_ids: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    likely_quality_band: str = Field(default="")
    common_strengths: list[str] = Field(default_factory=list)
    common_weaknesses: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)


class CrossSectionalStructure(BaseModel):
    """Cross-sectional clustering structure for one question part."""

    question_id: str
    part_id: str
    n_answers: int = Field(..., ge=0)
    clusters: list[ClusterSummary] = Field(default_factory=list)
    outliers: list[str] = Field(default_factory=list, description="submission_ids flagged as outliers")
    borderline_cases: list[str] = Field(default_factory=list, description="submission_ids near cluster boundaries")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Score awarded for a single criterion."""

    criterion_id: str
    score: float = Field(..., ge=0.0)
    max_score: float = Field(..., ge=0.0)
    evidence: str = Field(default="", description="Quoted or paraphrased evidence from the answer")
    reason: str = Field(default="", description="Justification for the score awarded")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def score_le_max(self) -> "CriterionScore":
        if self.score > self.max_score:
            raise ValueError(
                f"criterion {self.criterion_id}: score ({self.score}) exceeds max_score ({self.max_score})"
            )
        return self


class ScoredAnswer(BaseModel):
    """Full scoring result for one answer part."""

    submission_id: str
    question_id: str
    part_id: str
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    raw_total: float = Field(default=0.0, ge=0.0)
    max_total: float = Field(default=0.0, ge=0.0)
    proposed_bucket: str = Field(default="F")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = Field(default=False)
    review_reason: str = Field(default="")

    @model_validator(mode="after")
    def total_le_max(self) -> "ScoredAnswer":
        if self.raw_total > self.max_total and self.max_total > 0:
            raise ValueError(
                f"raw_total ({self.raw_total}) exceeds max_total ({self.max_total}) for {self.submission_id}"
            )
        return self


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class CalibratedAnswer(BaseModel):
    """Calibration result for one answer part."""

    submission_id: str
    anonymised_student_id: str = Field(default="")
    question_id: str
    part_id: str
    initial_score: float = Field(..., ge=0.0)
    calibrated_score: float = Field(..., ge=0.0)
    max_score: float = Field(default=0.0, ge=0.0)
    initial_bucket: str
    calibrated_bucket: str
    calibration_notes: str = Field(default="")
    nearest_neighbour_consistency: str = Field(default="")
    pairwise_checks: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class FeedbackContent(BaseModel):
    """Structured feedback content."""

    strengths: str = Field(default="")
    limitations: str = Field(default="")
    improvement_advice: str = Field(default="")
    summary: str = Field(default="")


class Feedback(BaseModel):
    """Complete feedback record for one answer part."""

    submission_id: str
    question_id: str
    part_id: str
    score: float = Field(..., ge=0.0)
    max_score: float = Field(..., ge=0.0)
    bucket: str
    feedback: FeedbackContent = Field(default_factory=FeedbackContent)
    evidence_used: list[str] = Field(default_factory=list)
    model_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human_review: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------


class HumanReviewEntry(BaseModel):
    """A record of a human reviewer's decision on one AI-marked answer."""

    review_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str
    reviewer: str
    submission_id: str
    question_id: str
    part_id: str
    ai_score: float
    human_score: float | None = None
    human_criterion_scores: dict[str, float] = Field(default_factory=dict)
    ai_bucket: str
    human_bucket: str | None = None
    ai_feedback: str = Field(default="")
    human_feedback: str | None = None
    review_action: str = Field(
        default="pending",
        description="One of: approved, edited, flagged_second_marker, flagged_extraction_error, flagged_rubric_ambiguity, flagged_similarity_concern",
    )
    review_notes: str = Field(default="")
    change_reason: str = Field(default="")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """Immutable record of a single LLM call for audit purposes."""

    timestamp: str
    stage: str
    submission_id: str | None = None
    question_id: str | None = None
    part_id: str | None = None
    model_name: str
    prompt_hash: str
    response_hash: str
    prompt_text: str | None = Field(default=None, description="Full local prompt text for audit reconstruction")
    response_text: str | None = Field(default=None, description="Full local model response for audit reconstruction")
    confidence: float | None = None
    notes: str = Field(default="")


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Model names for each pipeline stage."""

    analysis_model: str = "qwen2.5:7b"
    scoring_model: str = "qwen2.5:7b"
    feedback_model: str = "gemma3:4b"
    embedding_model: str = "nomic-embed-text"
    reranker_model: str | None = None


class OllamaConfig(BaseModel):
    """Ollama server connection settings."""

    base_url: str = "http://localhost:11434"
    timeout: int = 120


class QuestionSplittingConfig(BaseModel):
    """Settings for splitting submission text into question parts."""

    method: Literal["heuristic", "llm", "hybrid"] = "heuristic"
    patterns: list[str] = Field(
        default_factory=lambda: [
            r"Question \d+",
            r"Part [a-z]",
            r"Q\d+[a-z]?",
        ]
    )
    llm_fallback_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Full pipeline configuration loaded from YAML."""

    assessment_name: str = "Unnamed Assessment"
    submissions_folder: str
    output_folder: str
    marking_scheme_path: str
    rubric_path: str | None = None
    instructions_path: str | None = None
    sample_answers_path: str | None = None
    question_splitting: QuestionSplittingConfig = Field(default_factory=QuestionSplittingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    batch_size: int = Field(default=10, ge=1)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    run_clustering: bool = True
    run_umap: bool = True
    run_pairwise_boundary: bool = True
    export_format: str = "xlsx"
