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
