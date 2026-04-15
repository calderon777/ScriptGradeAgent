# Execution Plan

Ranked from highest priority to lowest priority.

Current extraction note:
- PDF handling is now hybrid by design.
- `PyMuPDF4LLM` cleaned text is used for structure detection only.
- The current extractor output is used for segmentation and scoring text because it preserves math and equations better.
- The two texts are not merged into one document; they are used at different pipeline stages.
- If future work changes PDF extraction again, preserve this split unless a replacement backend clearly beats the current extractor on both structure recovery and equation retention in end-to-end tests.

Next steps in this area:
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
