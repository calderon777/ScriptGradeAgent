# Execution Plan

This file is the repo's TOWS plan. It converts current SWOT forces into milestone and issue candidates, then orders them into daily execution steps.

Read the working docs in this order:

1. [small_model_scoring_payload_design.md](small_model_scoring_payload_design.md)
   - payload policy
2. [model_runtime_guidance.md](model_runtime_guidance.md)
   - runtime and inference policy
3. this file
   - work order, milestone ladder, and issue queue

## Current Repo Snapshot

Observed on 2026-04-16:

- Canonical app entrypoint is `app.py`.
- CI already exists in `.github/workflows/ci.yml`.
- Local validation passed:
  - `python -m unittest discover -s tests` -> 107 tests passed
  - `python -m compileall app.py batch_mark.py mvp_mark_one.py marking_pipeline tests` -> passed
- The repo has strong inspection tooling:
  - `scripts/run_human_benchmark.py`
  - `scripts/compare_assessment_structure_methods.py`
  - `scripts/export_assessment_structure_inspection.py`
  - `scripts/export_scoring_payload_inspection.py`
- The main engineering hotspot is still concentrated in:
  - `marking_pipeline/core.py` at about 5,022 lines
  - `tests/test_core.py` at about 2,291 lines
- Current benchmark evidence from `output/human_benchmark_ec3040/summary.json`:
  - mean absolute total-mark delta: `9.38`
  - PDF mean absolute total-mark delta: `13.0`
  - DOCX mean absolute total-mark delta: `5.75`
  - mean total latency: `373.7s`
- Current structure evidence from `output/assessment_structure_bakeoff/assessment_structure_bakeoff.json`:
  - deterministic structure extraction is ahead of the old heuristic baseline
- Current DOCX evidence from `output/docx_competitor_smoke_10/summary.json`:
  - the current backend can hit exact top-level part matches, but not consistently enough to call the problem solved

## Repo SWOT

### Strengths

1. The repo already has a coherent governance stack:
   - `constitution.md`
   - payload design doc
   - runtime guidance doc
   - execution plan
2. The pipeline is audit-oriented:
   - deterministic totals
   - prepared assessment maps
   - verifier hooks
   - benchmark and export scripts
3. The test baseline is real, not nominal:
   - 107 passing tests
   - many core behaviors already pinned
4. The repo already stores benchmark artifacts and comparison outputs, so changes can be evaluated against evidence rather than memory.
5. CI is already in place, so the next step is better regression content, not basic setup.

### Weaknesses

1. Accuracy is still not trustworthy enough on the fixed benchmark, especially for PDFs and deterministic derivation sections.
2. `marking_pipeline/core.py` is too large, which raises change risk and slows focused iteration.
3. `tests/test_core.py` is also concentrated into one large file, which makes failures harder to localize.
4. Benchmark scripts still depend on machine-specific paths and local artifacts outside the repo, which weakens portability.
5. Moderation is consuming latency even when it produces no mark change.
6. The previous `execution_plan.md` had stale items:
   - CI was listed as future work even though it already exists
   - legacy app cleanup was listed even though those files are not in the current tree

### Opportunities

1. The existing benchmark harness can become a formal regression gate with very little new infrastructure.
2. The payload and structure inspection scripts can directly support issue-driven debugging.
3. DOCX performance is already materially better than PDF performance, which gives a usable reference path for cross-format comparisons.
4. Prepared assessment maps, cache logic, and artifact enrichment already exist and can be used to isolate real gains from prompt churn.
5. The repo already contains enough benchmark output to define numeric success criteria for the next sprint.

### Threats

1. Overfitting to EC3040-style prompts could improve one benchmark while weakening general assessment behavior.
2. PDF extraction damage can silently distort the actual task before scoring begins.
3. Latency at current levels can make the app operationally unattractive even when marks improve.
4. Hard-coded private data paths and real sample artifacts create portability and data-governance risk.
5. The monolithic core raises the chance that a local fix creates a regression elsewhere in extraction, scoring, or reporting.

## TOWS To Milestones

### SO Strategies

Use existing strengths to exploit current opportunities.

1. Turn benchmark scripts plus stored outputs into a formal regression milestone.
   - Milestone: `M4 Benchmark Productization`
2. Use passing tests and exported inspection artifacts to make issue triage evidence-first.
   - Milestone: `M1 Trustworthy Payloads`
   - Milestone: `M2 Scoring Calibration`

### ST Strategies

Use existing strengths to contain current threats.

1. Use the constitution, deterministic totals, and current tests to stop accuracy fixes from becoming silent logic drift.
   - Milestone: `M2 Scoring Calibration`
2. Use existing timing and benchmark outputs to cut latency only where value is unproven.
   - Milestone: `M3 Latency And Moderation Discipline`
3. Use the structure-comparison tooling to keep EC3040-specific fixes from becoming hidden general-policy changes.
   - Milestone: `M5 Structure Mismatch Visibility`

### WO Strategies

Use current opportunities to fix repo weaknesses.

1. Use the benchmark harness to replace vague planning with ranked issue tickets and measurable exit criteria.
   - Milestone: `M1 Trustworthy Payloads`
   - Milestone: `M2 Scoring Calibration`
2. Use existing CI and scripts to remove machine-specific benchmark friction.
   - Milestone: `M4 Benchmark Productization`
3. Use the current evidence base to split the monolith only after the highest-value quality issues are pinned.
   - Milestone: `M6 Architecture Hardening`

### WT Strategies

Reduce weaknesses that amplify external threats.

1. Reduce concentrated code risk before broader feature work such as OCR or new model routes.
   - Milestone: `M6 Architecture Hardening`
2. Clean up data and fixture policy before scaling benchmark usage.
   - Milestone: `M7 Data Hygiene`
3. Keep OCR in backlog until real workflow evidence justifies it.
   - Milestone: `M7 Data Hygiene`

## Milestone Ladder

Create milestones in this order.

### M1 Trustworthy Payloads

Priority: `P0`

Goal:
- Preserve exact task meaning for high-value sections before calibration starts.

Issue candidates:
1. `[P0][Payload] Stop aggressive truncation of exact question text for deterministic units`
2. `[P0][Payload] Detect damaged section wording and fall back to a larger local chunk`
3. `[P0][Payload] Export before/after payload inspections for PDF outlier sections`

Exit criteria:
- No `question_text_exact` payload for the targeted PDF outliers ends with a harmful ellipsis when the source contains more recoverable local text.
- The known `Part 2 Q2` and `Part 2 Q3` payloads preserve the actual section request well enough for manual inspection.

### M2 Scoring Calibration

Priority: `P0`

Goal:
- Reduce mark error on deterministic and PDF-heavy failures without hard-coding EC3040 topic language.

Issue candidates:
1. `[P0][Scoring] Tighten derivation scoring around inequality validity and family-by-family deviation logic`
2. `[P0][Scoring] Add missing-answer and low-evidence benchmark cases for zero or near-zero credit`
3. `[P0][Benchmark] Re-run the fixed 4-script benchmark after each scoring change`

Exit criteria:
- Improve `mean_abs_total_delta` from `9.38` to `<= 7.5`
- Improve PDF `mean_abs_total_delta` from `13.0` to `<= 9.0`
- Keep DOCX `mean_abs_total_delta` at `<= 6.0`

### M3 Latency And Moderation Discipline

Priority: `P1`

Goal:
- Remove time spent on moderation when it does not change outcomes.

Issue candidates:
1. `[P1][Latency] Log moderation trigger reasons and resulting mark deltas on the fixed benchmark`
2. `[P1][Latency] Narrow or disable moderation paths that repeatedly return zero change`
3. `[P1][Runtime] Compare quality before and after moderation narrowing on the same benchmark set`

Exit criteria:
- Reduce benchmark mean latency from `373.7s` to `< 320s`
- Do not worsen `mean_abs_total_delta` relative to the pre-change baseline

### M4 Benchmark Productization

Priority: `P1`

Goal:
- Make the benchmark process portable, reproducible, and issue-friendly.

Issue candidates:
1. `[P1][Benchmark] Remove hard-coded machine-specific defaults from benchmark scripts`
2. `[P1][Benchmark] Standardize benchmark artifact output paths inside the repo`
3. `[P1][Docs] Add one reproducible benchmark command per benchmark family`

Exit criteria:
- Benchmark scripts can run from repo-relative or CLI-supplied paths without editing source files.
- New benchmark outputs land in stable folders that are easy to diff across runs.

### M5 Structure Mismatch Visibility

Priority: `P1`

Goal:
- Surface disagreements between assessment-side and scheme-side structure before they become silent scoring defects.

Issue candidates:
1. `[P1][Structure] Export assessment-map vs marking-scheme mismatch reports`
2. `[P1][Structure] Define repair precedence rules for scheme-first vs assessment-side text`
3. `[P1][Inspection] Add mismatch examples to the inspection workflow`

Exit criteria:
- Mismatch cases are explicitly reported and inspectable.
- Repair rules are documented and reflected in exported artifacts.

### M6 Architecture Hardening

Priority: `P2`

Goal:
- Reduce change risk after the highest-value quality fixes are stabilized.

Issue candidates:
1. `[P2][Refactor] Split extraction, assessment-preparation, scoring, and moderation concerns out of core.py`
2. `[P2][Tests] Split test_core.py by subsystem`
3. `[P2][Docs] Document subsystem ownership and main regression hooks`

Exit criteria:
- `core.py` is materially smaller
- tests are grouped by subsystem
- no loss in test coverage or benchmark reproducibility

### M7 Data Hygiene

Priority: `P2`

Goal:
- Reduce data and portability risk before scaling the benchmark corpus.

Issue candidates:
1. `[P2][Data] Audit sample artifacts under scripts, output, and cache-backed workflows`
2. `[P2][Data] Define fixture policy for private grading data vs repo-safe fixtures`
3. `[P2][Backlog] Decide whether OCR remains backlog or becomes a planned feature`

Exit criteria:
- Sample-data policy is explicit
- sensitive or machine-bound fixtures are separated from repo-safe fixtures

## Ranked Daily Milestones

This is the next ranked daily sequence. Each day should close or materially advance one issue group.

### Day 1

Open the milestone and issue skeleton from `M1` and `M2`, then freeze the benchmark baseline.

Tasks:
- create the `M1` to `M7` milestone set
- create the `P0` issues first
- record the current benchmark numbers from `output/human_benchmark_ec3040/summary.json` as the baseline
- link each P0 issue to one concrete benchmark failure or exported artifact

Done when:
- the issue tracker reflects the same ordering as this file

### Day 2

Fix question-text truncation for deterministic units.

Tasks:
- inspect the exact payload builder path
- stop harmful ellipsis-based shortening for deterministic sections
- add tests around payload preservation
- export updated payload inspections for the known PDF outliers

Done when:
- targeted deterministic payloads preserve the real task wording on inspection

### Day 3

Add damaged-text detection and larger local chunk fallback.

Tasks:
- detect merged-word or visibly incomplete section wording
- switch to a larger local chunk only when the exact section looks damaged
- add tests for fallback activation and non-activation
- rerun the two PDF outlier scripts first

Done when:
- fallback behavior is selective and inspectable, not global

### Day 4

Tighten deterministic derivation scoring.

Tasks:
- sharpen criteria for formula validity, inequality logic, deviation checks, and numerical examples
- add missing-answer and low-evidence tests
- rerun the fixed 4-script benchmark

Done when:
- part-level deltas improve on the targeted derivation failures without adding topic-specific rules

### Day 5

Audit moderation value versus latency.

Tasks:
- log trigger reasons and delta outcomes for moderation
- identify no-op moderation cases
- narrow or disable no-op branches
- rerun the same benchmark to compare latency and quality

Done when:
- moderation has a measured justification or is removed from the affected path

### Day 6

Make structure mismatches visible.

Tasks:
- generate explicit mismatch reports between assessment-side and scheme-side structure
- document repair precedence
- add at least one regression test or inspection artifact for mismatch handling

Done when:
- structure disagreements are surfaced before scoring, not discovered after bad marks

### Day 7

Make the benchmark workflow portable.

Tasks:
- remove hard-coded OneDrive-style defaults from benchmark scripts where practical
- prefer CLI arguments, repo-relative fixtures, or documented local overrides
- standardize output locations for reruns

Done when:
- another machine can run the same scripts without editing source files

### Day 8

Reduce monolith risk after the quality path is stable.

Tasks:
- split `core.py` into smaller responsibility-focused modules
- split `tests/test_core.py` into matching test files
- keep public imports stable through `marking_pipeline/__init__.py`

Done when:
- the code layout is easier to reason about and tests still pass unchanged

### Day 9

Run the data-hygiene pass.

Tasks:
- audit repo-safe fixtures versus private benchmark data
- document what must stay local
- decide whether OCR is a backlog item or a planned milestone

Done when:
- data policy is explicit and future contributors can tell which artifacts are safe to keep in-repo

### Day 10

Close the sprint with a benchmarked release gate.

Tasks:
- rerun tests
- rerun the fixed benchmark
- compare against Day 1 baseline
- close completed issues and re-rank the remainder

Done when:
- the tracker shows which issues moved the benchmark and which did not

## Not Current Priorities

These should not be treated as current milestone work unless repo evidence changes.

1. Adding CI
   - already done in `.github/workflows/ci.yml`
2. Cleaning up `app_v1_mvp.py` and `app_v2_mvp.py`
   - those files are not in the current repo tree
3. Adding OCR immediately
   - keep as backlog until the data policy and benchmark goals justify it

## Decision Rule

For the next sprint, do not open new feature work until `M1`, `M2`, and `M3` are materially advanced. The repo's highest-value path is:

1. preserve the task correctly
2. score deterministic sections more truthfully
3. remove latency that is not buying quality
4. only then widen the benchmark and refactor the architecture
