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


def read_uploaded_files_text(uploaded_files):
    """Read and concatenate text from a list of uploaded files."""
    if not uploaded_files:
        return ""
    texts = []
    for f in uploaded_files:
        name = f.name.lower()
        if name.endswith(".txt"):
            texts.append(f.read().decode("utf-8", errors="ignore"))
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(f))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(f))
    return "\n\n".join(texts).strip()

MARK_DESCRIPTOR_SUMMARY = """
Use the following generic undergraduate mark descriptors as guidance (ignore the university name):

- 70–85 (First): responds fully to the task; shows excellent to outstanding knowledge, strong critical analysis, originality and insight; very well structured and well written.
- 60–69 (Upper Second / 2:1): responds to most criteria; thorough grasp of theory and concepts; good structure and clear argument; some critical analysis and synthesis.
- 50–59 (Lower Second / 2:2): generally effective response; clear grasp of main material; some organisation and analysis but limited depth or originality.
- 40–49 (Third): basic but acceptable response; covers core points with limited analysis; derivative and descriptive; weak conceptual grasp.
- 30–39: overall insufficient; significant gaps or inaccuracies; weakly developed understanding.
- 0–29: poor or very poor; little relevant knowledge or serious misunderstanding of the task.
"""

def call_ollama(
    text: str,
    rubric_text: str,
    brief_text: str,
    marking_scheme_text: str,
    graded_sample_text: str,
    other_context_text: str,
    filename: str,
    model_name: str,
) -> dict:
    """
    Use local LLM via Ollama to mark a script.
    The model should infer question structure and weights from the documents when possible.
    JSON schema:

    {
      "scale": "qualitative_0_85" or "quantitative_0_100",
      "total_mark": number,
      "max_mark": number,
      "questions": [
        {"id": "Q1", "mark": number, "max_mark": number, "feedback": "string"},
        ...
      ],
      "overall_feedback": "string"
    }
    """
    system = (
    "You are a fair and consistent university examiner. "
    "You strictly follow the rubric, assignment brief, marking scheme and example grading style when provided. "
    "Do NOT invent irrelevant content. "
    "Stay focused on the student's answer and the marking documents."
    )


    context_parts = [MARK_DESCRIPTOR_SUMMARY]

    if rubric_text:
        context_parts.append(f"RUBRIC:\n{rubric_text}")

    if brief_text:
        context_parts.append(f"ASSIGNMENT BRIEF:\n{brief_text}")

    if marking_scheme_text:
        context_parts.append(f"MARKING SCHEME:\n{marking_scheme_text}")

    if graded_sample_text:
        context_parts.append(
            "EXAMPLE GRADED SCRIPT (answer + mark + feedback to imitate):\n"
            f"{graded_sample_text}"
        )

    if other_context_text:
        context_parts.append(f"OTHER SUPPORTING DOCUMENTS:\n{other_context_text}")

    if context_parts:
        context = "\n\n".join(context_parts)
    else:
        # Fallback if no context uploaded at all
        context = (
            "No rubric or brief was provided. "
            "Mark primarily on economic understanding, use of theory, structure, and clarity."
        )

    user = f"""
You are marking a student's exam script in economics.

CONTEXT DOCUMENTS (to infer questions, criteria and weights):
{context}

STUDENT FILE NAME: {filename}

STUDENT ANSWER:
\"\"\"{text}\"\"\"

INSTRUCTIONS:

1. From the context documents, identify:
   - how many questions the exam has (e.g. two questions),
   - the mark allocation / weights for each question (e.g. Q1 40 marks, Q2 45 marks, total 85),
   - any explicit criteria for each question.

2. If the documents clearly specify a total mark and per-question marks, you MUST use those.
   For example, if the brief or marking scheme says "Q1 (40 marks), Q2 (45 marks), total 85",
   then Q1.max_mark = 40, Q2.max_mark = 45, total max_mark = 85.

3. If the documents do NOT clearly specify a total mark:
   - Decide whether the exam is QUALITATIVE (essay-style, discursive, open answers)
     or QUANTITATIVE (mathematical, clearly right/wrong).
   - If it is QUALITATIVE, use the DMU-style QUALITATIVE scale 0–85:
       scale = "qualitative_0_85"
       max_mark = 85
   - If it is clearly QUANTITATIVE, use scale 0–100:
       scale = "quantitative_0_100"
       max_mark = 100

4. Always:
   - Assign a mark for EACH QUESTION (field questions[i].mark) and a max_mark for that question.
   - Provide brief, focused feedback per question.
   - Provide a short overall feedback summary at the end (overall_feedback).

5. The overall total_mark must be the sum of the question marks
   and must not exceed max_mark.

Return ONLY valid JSON in this exact structure (no commentary, no markdown):

{{
  "scale": "qualitative_0_85",
  "total_mark": 68,
  "max_mark": 85,
  "questions": [
    {{
      "id": "Q1",
      "mark": 32,
      "max_mark": 40,
      "feedback": "Short feedback specific to question 1."
    }},
    {{
      "id": "Q2",
      "mark": 36,
      "max_mark": 45,
      "feedback": "Short feedback specific to question 2."
    }}
  ],
  "overall_feedback": "Short overall feedback for the whole script."
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

    # Safety: if model ignores the scale and returns a total_mark > max_mark but ≤ 100,
    # treat total_mark as a percentage and rescale to max_mark.
    if isinstance(data.get("total_mark"), (int, float)) and isinstance(data.get("max_mark"), (int, float)):
        if data["total_mark"] > data["max_mark"] and data["total_mark"] <= 100:
            original = float(data["total_mark"])
            max_mark = float(data["max_mark"])
            data["total_mark"] = round(original * max_mark / 100.0, 1)

    # If the model forgot max_mark, set it from the scale
    scale = data.get("scale")
    if "max_mark" not in data or data["max_mark"] in (None, 0):
        if scale == "qualitative_0_85":
            data["max_mark"] = 85
        elif scale == "quantitative_0_100":
            data["max_mark"] = 100


    return data


# ==== STREAMLIT APP ====


def main():
    st.set_page_config(page_title="St.Mark.GPT / ScriptGradeAgent", layout="wide")

    # Sidebar: name + status icon/text
    with st.sidebar:
        st.title("St.Mark.GPT")
        sidebar_status = st.empty()
        sidebar_status.info("📚 Ready to mark")

    st.title("St.Mark.GPT – Local AI Marking Assistant")

    st.markdown(
        "Upload student scripts (PDF or Word), optionally provide rubric / brief / marking scheme "
        "and example grading, and get an Excel file with marks and feedback from two local models."
    )

    # STEP 1 – Models (info only)
    st.header("Step 1 – Models (information)")
    st.markdown(
        "This app uses local models via Ollama. "
        "Please install Ollama and pull these models before running marking:"
    )
    for model_name, _ in MODELS:
        st.code(f"ollama pull {model_name}", language="bash")

    # STEP 2 – Upload scripts (required)
    st.header("Step 2 – Upload scripts (required)")
    st.caption("Upload one or more student scripts (.pdf or .docx).")
    script_files = st.file_uploader(
        "Drag and drop scripts here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    # STEP 3 – Additional marking context (optional)
    st.header("Step 3 – Additional marking context (optional)")

    st.subheader("3.1 Rubric (optional)")
    rubric_files = st.file_uploader(
        "Rubric files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="rubric_files",
    )

    st.subheader("3.2 Assignment brief (optional)")
    brief_files = st.file_uploader(
        "Assignment brief files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="brief_files",
    )

    st.subheader("3.3 Marking scheme (optional)")
    marking_scheme_files = st.file_uploader(
        "Marking scheme files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="marking_scheme_files",
    )

    st.subheader("3.4 Example graded script (optional)")
    graded_sample_files = st.file_uploader(
        "Example graded script files (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="graded_sample_files",
    )

    st.subheader("3.5 Other supporting files (optional)")
    other_files = st.file_uploader(
        "Other supporting documents (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        key="other_files",
    )

    # Turn uploaded context files into text
    rubric_text = read_uploaded_files_text(rubric_files)
    brief_text = read_uploaded_files_text(brief_files)
    marking_scheme_text = read_uploaded_files_text(marking_scheme_files)
    graded_sample_text = read_uploaded_files_text(graded_sample_files)
    other_context_text = read_uploaded_files_text(other_files)

    # STEP 4 – Run marking
    st.header("Step 4 – Run marking")

    if st.button("Run marking", type="primary"):
        if not script_files:
            st.error("Please upload at least one script in Step 2.")
            return

        sidebar_status.info("📖✏️ Marking in progress…")

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
                        marking_scheme_text,
                        graded_sample_text,
                        other_context_text,
                        filename,
                        model_name,
                    )
                    row[f"{label}_mark"] = result.get("total_mark")
                    row[f"{label}_feedback"] = result.get("overall_feedback") or result.get("feedback")
                except Exception as e:
                    row[f"{label}_mark"] = None
                    row[f"{label}_feedback"] = f"ERROR: {e}"

            rows.append(row)
            progress.progress(i / len(script_files))

        if not rows:
            st.warning("No results to show.")
            sidebar_status.info("📚 Ready to mark")
            return

        df = pd.DataFrame(rows)

        # Add comparison columns
        if "Gemma3_4B_mark" in df.columns and "Llama3.1_8B_mark" in df.columns:
            df["average_mark"] = df[["Gemma3_4B_mark", "Llama3.1_8B_mark"]].mean(axis=1)
            df["mark_difference"] = (
                df["Gemma3_4B_mark"] - df["Llama3.1_8B_mark"]
            ).abs()

        st.success("Marking completed.")
        sidebar_status.success("✅ Marking completed")

        st.subheader("Preview of results")
        st.dataframe(df, use_container_width=True)

        # Create Excel in memory
        output = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"St.Mark.GPT_results_{timestamp}.xlsx"
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
