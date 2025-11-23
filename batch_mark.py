import os
import re
import json
import pdfplumber
import requests
import pandas as pd
from datetime import datetime


# === CONFIG ===
BASE_DIR = r"C:\Users\Cam\Documents\GitProjects\ScriptGradeAgent"

SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts", "test")
RUBRIC_FILE = os.path.join(BASE_DIR, "rubrics", "labour_year2_rubric.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    ("gemma3:4b", "Gemma3_4B"),
    ("llama3.1:8b", "Llama3.1_8B"),
]


def list_pdfs(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]


def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()


def load_rubric(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_ollama(text, rubric, filename, model_name):
    system = "You are a fair university examiner."

    user = f"""
RUBRIC:
{rubric}

STUDENT FILE: {filename}

STUDENT ANSWER:
\"\"\"{text}\"\"\"

Return JSON only:
{{"total_mark": number, "max_mark": number, "feedback": "text"}}
"""

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    content = r.json()["message"]["content"]

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")

    return json.loads(match.group(0))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rubric = load_rubric(RUBRIC_FILE)
    files = list_pdfs(SCRIPTS_DIR)

    if not files:
        print("No PDFs found.")
        return

    rows = []

    for file in files:
        print(f"\nMarking: {file}")
        path = os.path.join(SCRIPTS_DIR, file)
        text = extract_text_from_pdf(path)

        row = {
            "filename": file
        }

        for model_name, label in MODELS:
            try:
                result = call_ollama(text, rubric, file, model_name)
                row[f"{label}_mark"] = result["total_mark"]
                row[f"{label}_feedback"] = result["feedback"]
                print(f"  {label}: {result['total_mark']}/{result['max_mark']}")
            except Exception as e:
                row[f"{label}_mark"] = None
                row[f"{label}_feedback"] = f"ERROR: {e}"

        rows.append(row)

    # Create dataframe
    df = pd.DataFrame(rows)

    # Add comparison columns
    if "Gemma3_4B_mark" in df and "Llama3.1_8B_mark" in df:
        df["average_mark"] = df[["Gemma3_4B_mark", "Llama3.1_8B_mark"]].mean(axis=1)
        df["mark_difference"] = (
            df["Gemma3_4B_mark"] - df["Llama3.1_8B_mark"]
        ).abs()

    # Save to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(OUTPUT_DIR, f"marks_{timestamp}.xlsx")
    df.to_excel(outfile, index=False)

    print(f"\nDone. Results saved to:\n{outfile}")


if __name__ == "__main__":
    main()
