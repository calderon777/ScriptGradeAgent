# ScriptGradeAgent Constitution

This repository exists to produce inspectable, auditable academic grading workflows for local models.

## Core Principles

1. Classification first
   Part or subpart classification is the primary grading act. Feedback is downstream of classification, not a substitute for it.

2. Deterministic totals
   The final mark is always computed by arithmetic from part or subpart scores. The system must not ask a model to invent the overall mark when deterministic component scores already exist.

3. Audit over cosmetics
   The pipeline must make errors visible. It must not hide grading failures behind vague prose, silent smoothing, or undocumented corrective logic.

4. Structure before language
   When an assessment provides part structure, weights, or numbering, the system must preserve that structure. Language generation must follow the grading structure rather than redefine it.

5. Evidence-bound grading
   Classification and scoring must be justified by evidence from the script and the supplied marking documents.

## Policies

### 1. Rubric Matrix Policy

The canonical rubric artifact in this repository is a rubric matrix, not a raw transcript of the marking scheme.

A rubric matrix must:
- be organized by part or subpart;
- state the unit max mark explicitly;
- state the grading mode explicitly;
- define ordered performance bands;
- define how low, mid, and high placement works within a band;
- summarize the core criterion being judged;
- support classification decisions directly.

Raw scheme text may be retained as source material, but it is not the final rubric format used for classification.

### 2. No Hidden Rescue Policy

Fallbacks may be used only for mechanical robustness, such as retrying malformed JSON or preserving diagnostic output. They must not silently alter grading logic, marks, or band placement without being surfaced in diagnostics.

The system must not use heuristic shortcuts that conceal underlying parsing, classification, or calibration problems.

### 3. Model Responsibility Policy

Models should be asked for the smallest decision that matters:
- detect structure;
- classify part quality;
- place work within a band;
- identify evidence.

The code should perform deterministic composition, export, and arithmetic whenever possible.

### 4. Benchmarking Policy

Any material grading change should be benchmarked against known human marks where available.

Benchmarks should report:
- success and failure rate;
- runtime;
- mark deltas against human marks;
- explicit validation notes;
- the exact rubric artifact used for the run.

### 5. Debug Policy

Debug mode is for exposing pipeline problems, not masking them.

In debug mode:
- raw model contract failures should remain visible;
- exported artifacts should preserve the exact rubric used;
- classification errors should be investigated at the part level before changing presentation logic.

### 6. General Assessment Policy

The grading pipeline must be written for general assessment use, not for a single sample assessment.

This means:
- task typing, criteria generation, and dependency logic should be driven by general instruction patterns and assessment constructs, not topic nouns from one course or exam;
- implementation choices should prefer cross-assessment validity over sample-specific benchmark gains;
- any sample used during development is evidence for failure modes, not a template to hard-code;
- when a heuristic is added, it should be justified in terms of assessment archetypes that can recur elsewhere;
- inspection exports and model-facing payloads should be checked against multiple task shapes whenever possible.
