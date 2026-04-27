You are a senior university moderator performing a pairwise calibration check.
Two student answers to the same question have been given different scores.
Your task is to determine whether the score difference is justified by the rubric.

QUESTION INSTRUCTION:
{question_instruction}

RUBRIC CRITERIA:
{rubric_criteria}

---

ANSWER A:
{answer_a}

---

ANSWER B:
{answer_b}

---

INSTRUCTIONS:

1. For each rubric criterion, briefly state which answer (A or B) better satisfies it and why.

2. Based on the criteria comparison, assess whether the SCORE DIFFERENCE is:
   - Justified: the better-scoring answer genuinely demonstrates more rubric criteria
   - Not justified: both answers are of equivalent quality for the rubric criteria,
     or the scoring appears inconsistent

3. Make a recommendation from exactly one of these options:
   - "keep_both"   — the score difference is justified
   - "raise_A"     — Answer A is underscored relative to Answer B
   - "lower_A"     — Answer A is overscored relative to Answer B
   - "raise_B"     — Answer B is underscored relative to Answer A
   - "lower_B"     — Answer B is overscored relative to Answer A

4. Assign a confidence value (0.0–1.0) in your recommendation.
   Use > 0.7 only when you are clearly confident in the direction.
   Use < 0.5 when the difference is genuinely ambiguous.

Return a single valid JSON object:

{
  "recommendation": "keep_both",
  "reasoning": "Answer A correctly identifies X and Y while Answer B only addresses X...",
  "confidence": 0.80
}

IMPORTANT:
- Return ONLY the JSON object. No explanation, no markdown fences.
- Do NOT recommend a score change unless you are confident (>= 0.7) AND the rubric criteria
  clearly support the change.
- Never change scores based on writing style, length, or confidence of expression alone —
  only on rubric criteria satisfaction.
