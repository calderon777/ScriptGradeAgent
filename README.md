# ScriptGradeAgent

ScriptGradeAgent is a local academic marking tool that reads submissions, applies uploaded marking documents, and sends the grading prompt to a local Ollama model.

## Canonical entrypoint

Use:

```bash
streamlit run app.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama and pull at least one supported model:

```bash
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2:7b
ollama pull gemma3:4b
```

Model/runtime guidance for this repo:

- [model_runtime_guidance.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/model_runtime_guidance.md)

## Working Documents

These documents are meant to be read together, in this order:

1. [small_model_scoring_payload_design.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/small_model_scoring_payload_design.md)
   - defines what information the small scoring model should receive
   - defines which payload families should exist across assessment types
2. [model_runtime_guidance.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/model_runtime_guidance.md)
   - defines how to run local Ollama models against those tasks
   - covers model roles, inference profiles, and failure investigation
3. [execution_plan.md](c:/Users/Cam/Documents/GitProjects/ScriptGradeAgent/execution_plan.md)
   - tracks the current engineering order of work
   - records the next benchmark and validation steps

Short rule:

- `small_model_scoring_payload_design.md` is payload policy
- `model_runtime_guidance.md` is model-usage policy
- `execution_plan.md` is implementation sequence

## Requirements for grading

- A rubric or marking scheme is required before grading starts.
- The uploaded marking documents must state the total mark explicitly, for example `Maximum mark: 20` or `out of 85`.
- OCR is not implemented yet. Scanned PDFs without extractable text are rejected instead of being graded.

## Batch scripts

- `batch_mark.py` marks every supported file under `scripts/test/` and writes an Excel file under `output/`.
- `mvp_mark_one.py` marks the first supported file under `scripts/test/`.

Both scripts use repo-relative paths and the same validation logic as the Streamlit app.

## Tests

Run:

```bash
python -m unittest discover -s tests
```
