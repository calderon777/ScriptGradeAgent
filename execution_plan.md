# Execution Plan

Ranked from highest priority to lowest priority.

Working-doc order:

1. [small_model_scoring_payload_design.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/small_model_scoring_payload_design.md)
   - payload policy
2. [model_runtime_guidance.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/model_runtime_guidance.md)
   - runtime and inference policy
3. this file
   - implementation and benchmark sequence

Current extraction note:
- PDF handling is now hybrid by design.
- `PyMuPDF4LLM` cleaned text is used for structure detection only.
- The current extractor output is used for segmentation and scoring text because it preserves math and equations better.
- The two texts are not merged into one document; they are used at different pipeline stages.
- If future work changes PDF extraction again, preserve this split unless a replacement backend clearly beats the current extractor on both structure recovery and equation retention in end-to-end tests.

Current structure note:
- Deterministic assessment-structure extraction is now the main baseline.
- The structure pass should preserve:
  - top-level parts
  - subparts, even when they are unmarked
  - exact task instructions
  - exact marking instructions
  - anchor phrases and evidence expectations
  - dependency links across parts
- Model tuning should now happen against this cleaner structure baseline rather than against the older heuristic-only structure path.
- Marking-scheme structure is now the canonical source when it is explicit enough.
- Assessment-side structure should now act as fallback or repair when the marking scheme is missing, header-only, or visibly damaged.

Current critical issues, ranked:

1. Exact section payload text is still degraded for some high-value units.
   - `Part 2 Q2` remains the clearest example.
   - The current payload still contains merged words such as `Deriveconditions...` and `inequilibrium`.
   - The scoring payload still trims `question.text` too aggressively and can end with `...`, which is risky for deterministic sections.
   - This is currently the highest-priority quality issue because it distorts the actual task before scoring begins.

2. The scorer still over-credits some deterministic derivation answers.
   - The one-script EC3040 smoke remains materially high on total-mark delta even after structure and profile changes.
   - `Part 2 Q2` is still being marked too generously relative to the human benchmark.
   - Required-step hints now exist, but the payload still does not force enough discrimination between:
     - presence of formulas
     - validity of inequalities
     - correct family-by-family deviation logic
     - valid numerical example

3. Larger local chunk fallback is not implemented yet.
   - The right policy is not minimalism at any cost.
   - When exact section wording looks damaged or incomplete, the system should send a slightly larger local chunk rather than a smaller but invalid one.
   - This is especially important for derivations, multi-step prompts, and sections with hidden subrequirements.

4. Disparities between assessment-map structure and marking-scheme structure still need a deliberate inspection pass.
   - The code now supports scheme-first canonical structure.
   - We still need a mismatch review between:
     - the assessment map built from assessment documents
     - the structure extracted from the marking scheme
   - This is important because downstream scoring quality will stay fragile if those two disagree silently.

5. Moderation is adding latency without clear value on the fixed smoke benchmark.
   - Stricter inference profiles made moderation eligible for `Part 2`, adding roughly 50 seconds.
   - That extra time has not produced a meaningful quality win on the fixed one-script benchmark.
   - The moderation gate needs inspection before further latency tuning.

6. The current scheme text is the right source, but still noisy.
   - The benchmark harness now uses marking-scheme-first context correctly.
   - However, the scheme extraction still contains merged words and truncated instruction lines.
   - So the remaining issue is no longer source precedence; it is extracted text cleanliness inside the chosen source.

7. Scoring payloads still contain some redundant or generic content.
   - The payloads are better than before, but they still repeat generic criterion language and weak/strong anchor patterns.
   - This affects latency and may reduce discrimination for small models.

Recommendations raised in this chat but not yet carried through:

- Stop truncating exact question text so aggressively for hard deterministic sections.
- Add a larger local chunk fallback when the extracted section wording looks damaged or incomplete.
- Add stricter derivation-specific criteria that test validity, not just apparent completeness.
- Inspect the moderation gate and disable or narrow it where it adds latency without quality gain.
- Run an explicit disparity review between the assessment map and the marking-scheme extraction artifact.
- Keep heuristics generic and avoid topic-specific overfitting.
- Continue using the deterministic Python section-detection fast path as the default where context structure is strong.

Next steps in this area:
- Immediate next move:
  - stop truncating exact question text so aggressively in the model-facing payload for deterministic sections
  - add a larger local chunk fallback when the extracted section wording looks damaged
  - rerun the fixed Sadik one-script benchmark after that change
- Next scoring-quality move after that:
  - sharpen deterministic derivation criteria so the scorer distinguishes valid conditions from merely complete-looking working
  - target `Part 2 Q2` and similar multi-step derivation questions without overfitting to EC3040 wording
- Immediate moderation move:
  - inspect why moderation is firing for `Part 2`
  - keep it only if it produces measurable quality improvement on the fixed benchmark
- Immediate structure-validation move:
  - perform an explicit disparity review between:
    - assessment map built from assessment-side documents
    - structure extracted from the marking scheme
  - record mismatches and decide when assessment-side text should repair scheme-side text
- Immediate scoring-payload efficiency move:
  - inspect the exported anchored payload workbook and identify repeated criterion or anchor wording across the 12 part calls
  - compress shared anchors into shorter status semantics or criterion phrasing without dropping task-specific criteria
  - rerun the same fixed 1-script benchmark after each compression step so quality and latency can be compared directly
- Promote this branch as the new baseline for ongoing work. The assessment-preparation cache refactor, Qwen/Qwen policy, PDF hybrid, and DOCX hybrid are a better default than the prior branch state.
- Run a small end-to-end benchmark against human marks for both PDF and DOCX:
  - reuse one prepared assessment map per assessment
  - grade at least 2 PDF submissions and 2 DOCX submissions
  - compare total mark, part-level marks, and feedback quality against human marking
- Focus the first post-merge DOCX check on the known risk:
  - confirm whether the adaptive DOCX scoring extractor reduces under-marking on the Aya sample and a second contrasting sample
  - if under-marking persists, inspect whether the loss comes from extraction, section detection, granularity refinement, or scoring prompts
- Validate the new DOCX heading hints on a broader mixed sample:
  - confirm they improve section detection often enough to justify keeping them
  - watch for false positives from bold inline prose or stylistic emphasis
- Keep the DOCX policy generic:
  - rely on formatting/layout signals first
  - avoid assessment-specific lexical heuristics
- Do not change the PDF split lightly:
  - structure text remains `PyMuPDF4LLM` with cleanup
  - scoring text remains the current extractor unless a new benchmark shows a better end-to-end outcome
- Record benchmark results in a stable output location so future extraction changes can be compared against the same cases.
- Recommended next benchmark slice after this extraction phase:
  - rerun the same 4-script EC3040 benchmark set after any scoring or missing-evidence changes
  - keep the benchmark fixed to:
    - Sadik PDF
    - Federica PDF
    - Aya DOCX
    - Apisan DOCX
  - target the next improvements at:
    - false positives for absent evidence
    - over-harsh per-part scoring on PDFs
    - alignment with human zero-credit decisions such as missing-answer cases
  - treat this as a scoring-calibration benchmark, not another extraction bakeoff, unless a later change clearly reopens extraction as the main bottleneck

1. Validate the canonical app end to end with real marking inputs
   Run `streamlit run app.py` and test the full workflow with:
   - a valid text-based submission and valid rubric
   - a missing rubric or marking scheme
   - a scanned or text-empty PDF
   Confirm that valid grading succeeds and invalid inputs fail clearly.

2. Verify Ollama model behavior against the stricter response contract
   Test each supported model (`llama3.1:8b`, `mistral:7b`, `qwen2:7b`, `gemma3:4b`) with representative submissions.
   Check for:
   - valid JSON output
   - correct `max_mark`
   - marks staying within range
   - feedback quality matching the rubric and script content

3. Tune prompts or narrow supported models based on actual failures
   If any model frequently returns invalid JSON, wrong scales, or weak feedback, either:
   - adjust the shared prompt logic in `marking_pipeline/core.py`, or
   - remove that model from the default supported list
   Do not keep unreliable models in the main path.

4. Decide whether the legacy entrypoints should remain in the repo
   `app_v1_mvp.py` and `app_v2_mvp.py` now forward to `app.py`.
   Keep them only if you still need compatibility with old launch commands; otherwise delete them to reduce ambiguity.

5. Confirm sample data policy and remove sensitive artifacts if needed
   Review `scripts/test/student1.pdf` and any remaining sample materials.
   If they are real assessment artifacts, remove them and clean repository history as needed.

6. Add an automated integration test for the shared grading flow
   Mock the Ollama API and cover:
   - a valid grading run
   - malformed model output
   - wrong mark scale from the model
   - empty extracted submission text

7. Add CI so the repo stays runnable
   Set up a simple CI workflow that runs:
   - dependency install
   - `python -m compileall`
   - `python -m unittest discover -s tests`

8. Improve user-facing error reporting in the app
   Replace raw `ERROR: ...` strings in result cells with cleaner structured statuses where useful.
   Keep technical detail available, but avoid mixing operational errors with normal grading output more than necessary.

9. Add OCR only if scanned PDFs are a real requirement
   The app now rejects scanned or text-empty PDFs instead of hallucinating marks.
   If scanned submissions are part of the real workflow, implement OCR as a deliberate next feature rather than a silent fallback.

10. Consider removing generated-output write paths from normal workflows
    The Streamlit app already returns in-memory downloads, which is cleaner.
    If batch scripts are mainly for local testing, consider whether writing Excel files to `output/` should stay as the default behavior.
