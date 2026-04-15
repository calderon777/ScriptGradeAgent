# Small-Model Scoring Payload Design

## Goal

Design the minimum scoring payload a small model needs in order to:

1. understand what the student was asked to do,
2. evaluate how well the response satisfies that task,
3. convert that judgement into a score within the available marks,
4. justify the score with short evidence.

The design should generalize across different assessment shapes. `EC3040` is used as a worked example of a sectioned multi-part exam. `Econ2545 rubric 2020.xlsx` is used as a contrasting example of a holistic criterion-matrix rubric. The code should not be overfit to either one.

## Design Principles

1. Preserve the construct being measured.
   - Do not collapse a discussion/synthesis task into a derivation task.
   - Do not replace exact question meaning with generic phrases like "the required analytical step".

2. Minimize reasoning load for small models.
   - One section at a time.
   - One scoring method per call.
   - Only the information needed for that section.

3. Prefer structured scoring instruments over prose rubric essays.
   - Short task definition.
   - 3 to 5 concrete criteria.
   - Explicit score logic.
   - Short output contract.

4. Separate human-inspection artifacts from model-facing artifacts.
   - Human-readable rubric matrices can be verbose.
   - Small-model prompts should be compressed and operational.

5. Keep prompt fields because they change the decision, not because they are nice to have.

## What The Model Should Decide

For each section, the model should answer only these questions:

1. What task was the student supposed to do?
2. What evidence in the response is relevant to that task?
3. How strong is the response on the section-specific scoring criteria?
4. What score follows from that judgement?

If a prompt field does not improve one of those decisions, it should not be sent.

## Canonical Input Blocks

Every section payload should be built from these blocks.

### 1. Section Metadata

Purpose:
- tie the scoring decision to the right section,
- enforce score range,
- carry only essential dependency information.

Fields:
- `section_id`
- `max_mark`
- `dependency_note` (only if the section explicitly depends on earlier sections)

### 2. Exact Question

Purpose:
- preserve the real task,
- prevent heuristic drift from the original exam wording.

Fields:
- `question_text_exact`
- `task_type`
- `task_goal`

`question_text_exact` should be the closest faithful extraction of the original section wording.

`task_type` should come from a controlled enum, not free prose.

Suggested enum:
- `deterministic_derivation`
- `explanation_interpretation`
- `comparative_statics`
- `welfare_reasoning`
- `evaluative_discussion`
- `critique_and_revision`
- `prediction_generation`
- `measurement_and_data_design`
- `regression_specification`
- `causal_identification`
- `synthesis_across_sources`
- `holistic_matrix_criterion`

### 3. Scoring Method

Purpose:
- tell the model what kind of judgement it is making.

Recommended methods:
- `criterion_evaluation`
- `classification_then_score`
- `criterion_row_scoring`

Avoid using:
- `agreement` for primary grading
- `frequency`, `likelihood`, `importance` unless the task itself genuinely asks for those constructs

### 4. Criteria

Purpose:
- operationalize what counts.

Each criterion should be:
- task-specific,
- concrete,
- short,
- observable from the response.

Recommended size:
- 3 to 5 criteria

Criterion fields:
- `criterion_name`
- `check`
- `weak_anchor` (optional)
- `strong_anchor` (optional)

### 5. Student Response

Purpose:
- provide only the text needed to score the section.

Fields:
- `student_response_text`

The model should not receive unrelated sections once the section boundary is fixed.

### 6. Output Contract

Purpose:
- constrain the model to a simple decision.

Recommended outputs:
- `provisional_score`
- `criterion_notes`
- `strengths`
- `weaknesses`
- `evidence`

## What Not To Send

Do not send these by default to a small model:

- whole-assessment structure hints once the section label is fixed,
- unrelated question texts,
- long rubric paragraphs,
- repeated generic examiner prose,
- multiple overlapping versions of the same criterion,
- background text that does not change the section judgement,
- neutral scoring options unless the section is genuinely missing or unscorable.

## Prompt Wording Recommendations

### Use `criterion_evaluation` when:
- the section has concrete required steps,
- partial credit matters,
- correctness can be decomposed.

Typical cases:
- derivation
- model setup
- variable definition
- regression specification
- endogeneity explanation with remedies

### Use `classification_then_score` when:
- the section is primarily qualitative or synthetic,
- the key decision is overall response quality.

Typical cases:
- evaluative discussion
- critique and revision
- synthesis across theory/evidence/sources

### Use `criterion_row_scoring` when:
- the rubric is already a matrix with criterion rows and ordered performance columns.

Typical case:
- `Econ2545 rubric 2020.xlsx`

The model should score each row against the existing rubric row rather than receive a rewritten generic rubric paragraph.

## Scale Structure Recommendations

### Recommended Structures

#### 1. Unipolar ordered scale

Use for:
- deterministic questions
- explanation questions
- method questions

Meaning:
- absent -> weak -> partial -> secure -> strong

#### 2. Forced-choice ordered bands

Use for:
- discursive or synthetic responses

Meaning:
- choose one quality band first, then place within that band

### Usually Avoid

#### Bipolar scales

Not appropriate for most exam scoring. Exam quality usually progresses from absent to strong, not between two opposite poles.

#### Neutral optional

Only allow a neutral / insufficient-evidence option when:
- the section text is missing,
- the extraction is clearly broken,
- the response does not address the section at all.

Otherwise it encourages non-committal answers.

## General Payload Templates

### Template A: Sectioned Exam, Deterministic / Structured

```json
{
  "section_id": "Part 2 Q3",
  "max_mark": 7.5,
  "question": {
    "text_exact": "...",
    "task_type": "deterministic_derivation",
    "task_goal": "derive the equilibrium conditions for scenario 2 and explain why they imply that scenario"
  },
  "scoring": {
    "method": "criterion_evaluation",
    "scale_structure": "unipolar",
    "criteria": [
      {"criterion_name": "setup", "check": "Uses the correct model setup and family choices relevant to the scenario."},
      {"criterion_name": "derivation", "check": "Derives the required inequalities or conditions correctly."},
      {"criterion_name": "reasoning", "check": "Shows enough intermediate reasoning to justify why the scenario follows."},
      {"criterion_name": "example", "check": "Provides a valid numerical example if the question asks for one."}
    ],
    "partial_credit_rule": "Award credit for correct setup and intermediate reasoning even if the final conditions are incomplete."
  },
  "student_response": {
    "text": "..."
  },
  "output_contract": {
    "return_fields": [
      "provisional_score",
      "criterion_notes",
      "strengths",
      "weaknesses",
      "evidence"
    ]
  }
}
```

### Template B: Sectioned Exam, Discursive / Synthetic

```json
{
  "section_id": "Part 4",
  "max_mark": 15,
  "question": {
    "text_exact": "...",
    "task_type": "synthesis_across_sources",
    "task_goal": "use the model and the empirical design to evaluate the claims made in the videos and identify points missing from the model"
  },
  "scoring": {
    "method": "classification_then_score",
    "scale_structure": "forced_choice_band",
    "bands": ["weak", "adequate", "strong", "excellent"],
    "criteria": [
      {"criterion_name": "claim_identification", "check": "Identifies concrete claims from the source material."},
      {"criterion_name": "model_application", "check": "Uses relevant elements of the theoretical model to support or challenge those claims."},
      {"criterion_name": "evidence_integration", "check": "Uses empirical design or evidence from Part 3 where relevant."},
      {"criterion_name": "extension", "check": "Explains what important points are missing from the model and how the model could be extended."}
    ]
  },
  "student_response": {
    "text": "..."
  },
  "output_contract": {
    "return_fields": [
      "quality_band",
      "within_band_position",
      "provisional_score",
      "strengths",
      "weaknesses",
      "evidence"
    ]
  }
}
```

### Template C: Holistic Matrix Rubric

Use when the source rubric already has meaningful criterion rows and ordered quality columns, as in `Econ2545 rubric 2020.xlsx`.

```json
{
  "criterion_id": "Subject knowledge",
  "weight": 0.20,
  "question_context": {
    "assignment_type": "holistic presentation / report rubric"
  },
  "scoring": {
    "method": "criterion_row_scoring",
    "scale_structure": "forced_choice_band",
    "bands": [
      {"name": "unsatisfactory", "descriptor": "..."},
      {"name": "weak", "descriptor": "..."},
      {"name": "satisfactory", "descriptor": "..."},
      {"name": "very_good", "descriptor": "..."},
      {"name": "outstanding", "descriptor": "..."}
    ]
  },
  "student_response": {
    "text": "..."
  }
}
```

For this rubric type, do **not** rewrite the row into generic criteria if the original row descriptors are already serviceable.

## Worked Mapping: EC3040

This table is a design aid, not a coding contract. The important point is the task type and minimal scoring payload, not the specific labels.

| Section | True task type | What the model must decide | Recommended method | Minimum information to send | What not to send |
| --- | --- | --- | --- | --- | --- |
| Part 1 | `critique_and_revision` | Whether the student generated, audited, and corrected an AI summary using course/report evidence | `classification_then_score` or hybrid with criteria | exact task text; explicit required four-part answer structure; criteria for accuracy checking, source use, page references, correction quality, rewrite quality; section text | unrelated Part 2–4 structure hints |
| Part 2 Q1 | `explanation_interpretation` | Whether payoff functions are correctly stated and interpreted term by term | `criterion_evaluation` | question text; criteria for formula correctness, term interpretation, connection to policy costs/benefits; section text | UK band prose |
| Part 2 Q2 | `deterministic_derivation` | Whether scenario 1 conditions are correctly derived and justified | `criterion_evaluation` | exact scenario wording; criteria for setup, inequalities, justification, numerical example; partial-credit rule; section text | generic analytical discussion language |
| Part 2 Q3 | `deterministic_derivation` | Whether scenario 2 conditions are correctly derived and justified | `criterion_evaluation` | same pattern as Q2 but for scenario 2 | generic "required analytical step" language |
| Part 2 Q4 | `comparative_statics` | Whether the student explains how parameter changes shift likelihood of scenario 1 vs scenario 2 | `criterion_evaluation` | question text; criteria for parameter-direction reasoning, scenario comparison, explanation quality; section text | long band descriptors |
| Part 2 Q5 | `welfare_reasoning` | Whether the student identifies when VAT raises or lowers welfare and tracks family-level effects | `criterion_evaluation` | question text; criteria for welfare conditions, sign logic, family-specific utility effects, completeness of reasoning; section text | unrelated marks hints |
| Part 2 Q6 | `evaluative_discussion` | Whether the student makes a justified judgement about VAT using model results and lecture concepts | `classification_then_score` | exact question text; criteria for stance, use of model results, intuition, distributional reasoning, lecture concepts; section text | deterministic partial-credit wording |
| Part 3 Q1 | `prediction_generation` | Whether the student states valid testable predictions implied by the model | `criterion_evaluation` | question text; criteria for validity, directionality, conditional predictions if relevant; section text | generalized evidence/appraisal prose |
| Part 3 Q2 | `measurement_and_data_design` | Whether the student operationalizes variables and identifies plausible data sources | `criterion_evaluation` | question text; criteria for variable measurement, source relevance, mapping from dataset to concept; section text | forced band language |
| Part 3 Q3 | `regression_specification` | Whether the student proposes a coherent regression, defines terms, and states the hypothesis | `criterion_evaluation` | question text; criteria for equation correctness, variable definitions, coefficient of interest, hypothesis statement; section text | whole-part structure hints |
| Part 3 Q4 | `causal_identification` | Whether the student explains endogeneity risks and feasible remedies | `criterion_evaluation` | question text; criteria for threat identification, link to theory, remedy quality, practical feasibility; section text | derivation-only framing |
| Part 4 | `synthesis_across_sources` | Whether the student uses model + evidence to assess video claims and propose model extensions | `classification_then_score` | exact question text; dependency note linking to Part 2 and Part 3; criteria for claim identification, theory application, evidence use, omitted arguments, extension proposals; section text | deterministic derivation rubric |

## What EC3040 Shows Without Overfitting

The EC3040 sample reveals general failure modes that can happen elsewhere:

1. A section can have correct marks and labels but the wrong task type.
2. Exact question meaning can be lost if the system summarizes too aggressively.
3. Generic criteria can become detached from the real thing being graded.
4. Small models are penalized if each section prompt contains unrelated context.

These are not EC3040-specific problems. They are general prompt-design risks.

## What Econ2545 Adds

`Econ2545 rubric 2020.xlsx` shows a different assessment shape:

- criterion rows already exist,
- ordered quality descriptors already exist,
- weights are explicit,
- a matrix is the native scoring form.

That means the best model-facing representation is probably:

1. one criterion row at a time, or
2. a small batch of criterion rows with their ordered descriptors.

The system should therefore support at least two prompt families:

1. `sectioned_exam_payload`
2. `criterion_matrix_payload`

Trying to force both through the same generic rubric prose generator will reduce validity.

## Recommended Next Implementation Steps

1. Preserve `question_text_exact` per section.
2. Replace heuristic-only `grading_mode` with a richer `task_type`.
3. Generate model-facing criteria from `task_type` plus exact question text.
4. Build different prompt constructors for:
   - deterministic structured tasks,
   - discursive / synthetic tasks,
   - criterion-matrix tasks.
5. Remove unrelated whole-assessment context from section scoring prompts.
6. Keep human-inspection exports, but separate them from model-facing payloads.

## Success Criteria

The redesigned payload is better if:

1. the model can identify the exact section task without inferring hidden intent,
2. prompts are materially shorter,
3. criteria are concrete and section-specific,
4. the scoring method matches the construct,
5. the rubric is faithful to the original question,
6. different assessment archetypes can use different payload builders.
