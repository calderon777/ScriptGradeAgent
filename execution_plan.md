# Execution Plan

Ranked from highest priority to lowest priority.

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
