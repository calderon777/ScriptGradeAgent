You are an expert university marker. Score the following student answer against each rubric criterion.
Work through the steps below IN ORDER. Do not skip steps.

QUESTION INSTRUCTION:
{question_instruction}

RUBRIC CRITERIA:
{rubric_criteria}

CRITERION DEFINITIONS:
{criterion_definitions}

STRUCTURAL ANALYSIS (from previous stage):
{structural_analysis}

BUCKET DEFINITIONS:
{bucket_definitions}

STUDENT ANSWER:
{student_answer}

---

STEP 1 — EVIDENCE EXTRACTION
For each criterion, copy or closely paraphrase the specific part of the student answer that
constitutes evidence for that criterion. If no evidence exists, write "No evidence found."

STEP 2 — CRITERION SCORING
For each criterion:
  a) State the evidence you found (from Step 1)
  b) Apply the partial credit rules and no-credit conditions from the rubric
  c) Assign a score (must be between 0 and the criterion's max_marks, inclusive)
  d) Write a one-sentence justification
  e) Assign a per-criterion confidence (0.0–1.0)

STEP 3 — TOTAL AND BUCKET
  a) Sum all criterion scores to produce raw_total
  b) raw_total must not exceed max_total
  c) Assign proposed_bucket based on the bucket_definitions above
     (the bucket whose mark_range contains raw_total)

STEP 4 — OVERALL CONFIDENCE
  Assign a single confidence value (0.0–1.0) reflecting your certainty in the overall score.
  Set needs_human_review=true if confidence < 0.65, if the answer is ambiguous,
  or if a criterion was very hard to score.

STEP 5 — RETURN JSON
Return a single valid JSON object:

{
  "criterion_scores": [
    {
      "criterion_id": "c1",
      "score": 3.0,
      "max_score": 4.0,
      "evidence": "The student states '...' which demonstrates...",
      "reason": "Partial credit awarded because...",
      "confidence": 0.85
    }
  ],
  "raw_total": 6.5,
  "max_total": 10.0,
  "proposed_bucket": "B",
  "confidence": 0.80,
  "needs_human_review": false,
  "review_reason": ""
}

IMPORTANT RULES:
- Return ONLY the JSON object. No explanation, no markdown fences.
- score must NEVER exceed max_score for any criterion.
- raw_total must NEVER exceed max_total.
- proposed_bucket MUST be one of A, B, C, D, E, F.
- Do not include submission_id, question_id, or part_id in the JSON.
- Do not award marks for content not present in the student answer.
- Do not penalise for style, length, or vocabulary unless the rubric explicitly requires it.
