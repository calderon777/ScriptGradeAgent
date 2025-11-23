import os
import re
import json
import pdfplumber
import requests


# Base folder for ScriptGradeAgent
BASE_DIR = r"C:\Users\Cam\Documents\GitProjects\ScriptGradeAgent"

SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts", "test")
RUBRIC_FILE = os.path.join(BASE_DIR, "rubrics", "labour_year2_rubric.txt")

OLLAMA_URL = "http://localhost:11434/api/chat"

# (model_name_in_ollama, nice_label_for_printing)
MODELS = [
    ("gemma3:4b", "Gemma3_4B"),
    ("llama3.1:8b", "Llama3.1_8B"),
]


def find_first_pdf(folder: str) -> str:
    """Return the full path of the first .pdf file in the folder."""
    for f in os.listdir(folder):
        if f.lower().endswith(".pdf"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No PDF found in {folder}")


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a digital (non-scanned) PDF using pdfplumber."""
    print(f"Reading PDF: {os.path.basename(path)}")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()


def load_rubric(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_ollama(text: str, rubric: str, filename: str, model_name: str) -> dict:
    """
    Send rubric + student answer to local LLM via Ollama.
    Expect JSON with: total_mark, max_mark, feedback.
    """
    system = "You are a fair and consistent university examiner in economics."
    user = f"""
You are marking a student's script.

RUBRIC:
{rubric}

STUDENT FILE: {filename}

STUDENT ANSWER:
\"\"\"{text}\"\"\"

TASK:
1. Assign a single overall mark (field "total_mark").
2. Repeat the maximum mark from the rubric as "max_mark".
3. Give concise feedback (field "feedback") in no more than 200 words.

Return ONLY valid JSON in this exact format:
{{
  "total_mark": 15,
  "max_mark": 20,
  "feedback": "Your answer shows good understanding of..."
}}
"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }

    print(f"Sending to local model: {model_name} ...")
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    content = r.json()["message"]["content"]

    # Find JSON object in the response (in case it adds ```json fences)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{content}")
    return json.loads(match.group(0))


def main():
    pdf_path = find_first_pdf(SCRIPTS_DIR)
    filename = os.path.basename(pdf_path)

    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("No text extracted. For the MVP, use a digital (non-scanned) PDF.")
        return

    rubric = load_rubric(RUBRIC_FILE)

    print(f"\n=== MARKING {filename} WITH MULTIPLE MODELS ===")

    for model_name, label in MODELS:
        try:
            result = call_ollama(text, rubric, filename, model_name)
            print(f"\n--- {label} ---")
            print(f"Mark: {result['total_mark']}/{result['max_mark']}")
            print("\nFeedback:")
            print(result["feedback"])
            print("------------------------")
        except Exception as e:
            print(f"\n--- {label} ---")
            print(f"ERROR while using model {model_name}: {e}")
            print("------------------------")


if __name__ == "__main__":
    main()
