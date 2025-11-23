import os
import io
import re
import json
from datetime import datetime

import pdfplumber
from docx import Document
import pandas as pd
import requests
import streamlit as st


# ==== CONFIG ====
OLLAMA_URL = "http://localhost:11434/api/chat"

# (ollama_model_name, label_used_in_columns)
MODELS = [
    ("gemma3:4b", "Gemma3_4B"),
    ("llama3.1:8b", "Llama3.1_8B"),
]


# ==== HELPERS ====


def extract_text_from_pdf(file_obj) -> str:
    """Extract text from a digital PDF (no OCR)."""
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()


def extract_text_from_docx(file_obj) -> str:
    """Extract text from a .docx file."""
    doc = Document(file_obj)
    return "\n".join(p.text for p in doc.paragraphs).strip()


def call_ollama(
    text: str,
    rubric: str,
    brief: str,
    sample_grade_text: str,
    filename: str,
    model_name: str,
) -> dict:
    """
    Send rubric + student answer (plus optional contextual info) to local LLM via Ollama.
    Expect JSON with: total_mark, max_mark, feedback.
    """
    system = (
        "You are a fair and consistent university examiner. "
        "You strictly follow the rubric, assignment brief, and example grading style."
    )

    context_parts = [f"RUBRIC:\n{rubric}"]

    if brief:
        context_parts.append(f"ASSIGNMENT BRIEF / MARKING SCHEME:\n{brief}")

    if sample_grade_text:
        context_parts.append(
            "EXAMPLE GRADED SCRIPT (showing target marking style):\n"
            f"{sample_grade_text}"
        )

    context = "\n\n".join(context_parts)

    user = f"""
You are marking a student's script.

{context}

STUDENT FILE NAME: {filename}

STUDENT ANSWER:
\"\"\"{text}\"\"\"

TASK:
1. Assign a single overall mark in the field "total_mark".
   The maximum mark is 20. You MUST choose a number between 0 and 20 inclusive.
2. Set "max_mark" to 20.
3. Give concise, constructive feedback in the field "feedback" in no more than 200 words.
   Focus on strengths, weaknesses, and how to improve.

Return ONLY valid JSON in this exact format (no extra text, no explanation):

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

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    content = r.json()["message"]["content"]

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{content}")
    data = json.loads(match.group(0))

    # Safety: if model ignores the 0–20 instruction and returns > 20, rescale as percentage
    if isinstance(data.get("total_mark"), (int, float)) and data.get("total_mark", 0) > 20:
        original = float(data["total_mark"])
        data["total_mark"] = round(original * 20.0 / 100.0, 1)

    # Ensure max_mark is 20
    data["max_mark"] = 20

    return data


# ==== STREAMLIT APP ====


def main():
    st.set_page_config(page_title="3rdGradeGPT / ScriptGradeAgent", layout="wide")
    st.title("3rdGradeGPT – Local AI Marking Assistant")

    st.markdown(
        "Upload student scripts (PDF or Word), provide a rubric and optional brief "
        "and example grading, and get an Excel file with marks and feedback from two local models."
    )

    with st.sidebar:
        st.header("Step 1 – Models")
        st.write(
            "This app uses local models via Ollama. "
            "Make sure Ollama is running and you have pulled these models:"
        )
        for model_name, _ in MODELS:
            st.code(f"ollama pull {model_name}", language="bash")

    st.header("Step 2 – Upload scripts")
    script_files = st.file_uploader(
        "Upload one or more student scripts (.pdf or .docx)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.header("Step 3 – Rubric and brief")

    rubric_tab, brief_tab, sample_tab = st.tabs(
        ["Rubric", "Assignment brief / marking scheme", "Example graded script"]
    )

    with rubric_tab:
        rubric_uploaded = st.file_uploader(
            "Upload rubric file (txt, pdf, or docx) OR paste rubric below",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=False,
            key="rubric_file",
        )
        rubric_text = st.text_area("Rubric (if not uploading a file):", height=200)

    with brief_tab:
        brief_uploaded = st.file_uploader(
            "Upload assignment brief / marking scheme (txt, pdf, or docx) OR paste below",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=False,
            key="brief_file",
        )
        brief_text = st.text_area(
            "Assignment brief / marking scheme text:", height=200
        )

    with sample_tab:
        st.markdown(
            "Paste an example graded script (student answer + mark + feedback) "
            "to steer the style (optional but recommended)."
        )
        sample_grade_text = st.text_area(
            "Example graded script:",
            height=200,
            placeholder="Student answer...\n\nMark: 15/20\nFeedback: ...",
        )

    # Load text from uploaded rubric/brief if present
    def read_uploaded_text(uploaded_file):
        if uploaded_file is None:
            return ""
        name = uploaded_file.name.lower()
        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file)
        elif name.endswith(".docx"):
            return extract_text_from_docx(uploaded_file)
        return ""

    if rubric_uploaded is not None:
        rubric_text = read_uploaded_text(rubric_uploaded)
    if brief_uploaded is not None:
        brief_text = read_uploaded_text(brief_uploaded)

    st.header("Step 4 – Run marking")

    if st.button("Run marking", type="primary"):
        if not script_files:
            st.error("Please upload at least one script.")
            return
        if not rubric_text.strip():
            st.error("Please provide a rubric (upload a file or paste text).")
            return

        rows = []
        progress = st.progress(0)
        status = st.empty()

        for i, uploaded in enumerate(script_files, start=1):
            filename = uploaded.name
            status.text(f"Marking {filename} ({i}/{len(script_files)})...")

            # Extract text from script
            try:
                if filename.lower().endswith(".pdf"):
                    script_text = extract_text_from_pdf(uploaded)
                elif filename.lower().endswith(".docx"):
                    script_text = extract_text_from_docx(uploaded)
                else:
                    script_text = ""
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
                continue

            row = {"filename": filename}

            for model_name, label in MODELS:
                try:
                    result = call_ollama(
                        script_text,
                        rubric_text,
                        brief_text,
                        sample_grade_text,
                        filename,
                        model_name,
                    )
                    row[f"{label}_mark"] = result.get("total_mark")
                    row[f"{label}_feedback"] = result.get("feedback")
                except Exception as e:
                    row[f"{label}_mark"] = None
                    row[f"{label}_feedback"] = f"ERROR: {e}"

            rows.append(row)
            progress.progress(i / len(script_files))

        if not rows:
            st.warning("No results to show.")
            return

        df = pd.DataFrame(rows)

        # Add comparison columns
        if "Gemma3_4B_mark" in df.columns and "Llama3.1_8B_mark" in df.columns:
            df["average_mark"] = df[["Gemma3_4B_mark", "Llama3.1_8B_mark"]].mean(
                axis=1
            )
            df["mark_difference"] = (
                df["Gemma3_4B_mark"] - df["Llama3.1_8B_mark"]
            ).abs()

        st.success("Marking completed.")
        st.subheader("Preview of results")
        st.dataframe(df, use_container_width=True)

        # Create Excel in memory
        output = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"3rdGradeGPT_results_{timestamp}.xlsx"
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        output.seek(0)

        st.download_button(
            label="Download Excel file",
            data=output,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
