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

# Available models: key is what the user sees, value is (ollama_model_name, label_used_in_columns)
AVAILABLE_MODELS = {
    "Llama 3.1 8B (recommended – more rigorous)": ("llama3.1:8b", "Llama3.1_8B"),
    "Mistral 7B (strong reasoning)": ("mistral:7b", "Mistral_7B"),
    "Qwen2 7B (good JSON + structure)": ("qwen2:7b", "Qwen2_7B"),
    "Gemma 3 4B (faster, softer feedback)": ("gemma3:4b", "Gemma3_4B"),
}



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

    Expected JSON schema (overall only, no per-question marks required):

    {
      "scale": "qualitative_0_85" or "quantitative_0_100",
      "total_mark": number,
      "max_mark": number,
      "overall_feedback": "string (≥150 words, referring to each question)"
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

    instructions = """
INSTRUCTIONS:

1. From the context documents, identify:
   - how many questions the exam has (e.g. two questions),
   - the mark allocation / weights for each question (e.g. Q1 40 marks, Q2 45 marks, total 85),
   - any explicit criteria for each question.

2. If the documents clearly specify a total mark and per-question marks, you MUST use those
   to guide your judgment. However, your final output will only contain an overall mark
   for the whole script (no per-question breakdown in the JSON).

3. MARKING AND FEEDBACK

   - You MUST:
       * Assign a single overall numeric mark for the script: total_mark.
       * Use max_mark = 85, unless the context clearly specifies a different total mark.
       * Provide overall_feedback that is at least 150 words.

   - FEEDBACK QUALITY REQUIREMENTS:
       * overall_feedback MUST clearly state:
           - what the student did well (strengths),
           - what the student needs to improve,
           - how they can improve in future submissions (concrete advice).
       * If there is more than one question in the exam, overall_feedback MUST explicitly
         mention each question, using labels such as:
         "Regarding Q1, ...", "Regarding Q2, ...", etc.

4. OUTPUT FORMAT

   - You MUST return ONLY a single JSON object with exactly these keys:
       * "scale" (string),
       * "total_mark" (number),
       * "max_mark" (number),
       * "overall_feedback" (string).

   - Use this scale unless the context clearly specifies a 0–100 scheme:
       * scale = "qualitative_0_85"
       * max_mark = 85

   - All marks must be numeric (not strings).

   - Do NOT include any extra keys, explanations, or text outside this JSON object.

EXAMPLE OF THE REQUIRED JSON SHAPE (values here are placeholders only):

{
  "scale": "qualitative_0_85",
  "total_mark": 0,
  "max_mark": 85,
  "overall_feedback": "overall feedback for the whole script, explicitly mentioning each question."
}
"""

    user = (
        "You are marking a student's exam script in economics.\n\n"
        "CONTEXT DOCUMENTS (to infer questions, criteria and weights):\n"
        f"{context}\n\n"
        f"STUDENT FILE NAME: {filename}\n\n"
        "STUDENT ANSWER:\n"
        f"\"\"\"{text}\"\"\"\n\n"
        f"{instructions}"
    )

    payload = {
        "model": model_name,
        # If your Ollama build supports it, this forces pure-JSON replies:
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    content = r.json()["message"]["content"]

    # Expect pure JSON now
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nFull content:\n{content}")

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

    # Final sanity checks on total_mark
    if not isinstance(data.get("total_mark"), (int, float)):
        raise ValueError(f"Model did not return a valid numeric total_mark. JSON was: {data}")

    if isinstance(data.get("max_mark"), (int, float)) and data["max_mark"] > 0:
        if data["total_mark"] < 0:
            data["total_mark"] = 0
        elif data["total_mark"] > data["max_mark"]:
            data["total_mark"] = data["max_mark"]

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

    # STEP 1 – Models (information + selection)
    st.header("Step 1 – Models (information)")
    st.markdown(
        "This app uses local models via Ollama. "
        "Please install Ollama and pull at least the **recommended** model before running marking."
    )

    # Show pull commands for all available models
    for display_name, (model_name, _) in AVAILABLE_MODELS.items():
        st.markdown(f"**{display_name}**")
        st.code(f"ollama pull {model_name}", language="bash")

    st.markdown("---")
    st.subheader("Model selection")

    model_choices = list(AVAILABLE_MODELS.keys())

    selected_model_keys = st.multiselect(
        "Choose which models to run (default: Llama 3.1 8B only, for speed)",
        options=model_choices,
        default=["Llama 3.1 8B (recommended – more rigorous)"],
    )

    if not selected_model_keys:
        st.warning("Please select at least one model above before running the marking.")
        selected_models = []
    else:
        selected_models = [AVAILABLE_MODELS[k] for k in selected_model_keys]

    # STEP 2 – Upload scripts (required)

    st.header("Step 2 – Upload scripts (required)")
    st.caption("Upload one or more student scripts (.pdf or .docx).")
    script_files = st.file_uploader(
        "Drag and drop scripts here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.subheader("Or upload a CSV with text answers")
    st.caption("Upload a .csv file where each row is a student and columns contain their answers.")
    csv_file = st.file_uploader(
        "CSV file with one row per student",
        type=["csv"],
        key="csv_assessments",
    )

    df_csv_preview = None
    if csv_file is not None:
        # Try to read CSV with UTF-8, fall back to latin-1
        try:
            csv_file.seek(0)
            df_csv_preview = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            csv_file.seek(0)
            df_csv_preview = pd.read_csv(csv_file, encoding="latin-1")

        st.caption("Preview of CSV (first 5 rows)")
        st.dataframe(df_csv_preview.head(), use_container_width=True)

        cols = list(df_csv_preview.columns)
        default_idx = cols.index("Id") if "Id" in cols else 0
        csv_id_col = st.selectbox(
            "Column to use as ID / filename",
            options=cols,
            index=default_idx,
            key="csv_id_col",
        )

        default_text_cols = [c for c in cols if "Question" in c or df_csv_preview[c].dtype == object]
        if not default_text_cols:
            default_text_cols = cols

        csv_text_cols = st.multiselect(
            "Columns containing answers to mark",
            options=cols,
            default=default_text_cols,
            key="csv_text_cols",
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
        if (not script_files) and (csv_file is None):
            st.error("Please upload at least one script in Step 2, either PDF/DOCX or a CSV file.")
            return

        if not selected_models:
            st.error("No models selected in Step 1. Please select at least one model.")
            return

        sidebar_status.info("📖✏️ Marking in progress…")

        rows = []
        progress = st.progress(0)
        status = st.empty()

        # === Branch 1: CSV-based assessments ===
        if csv_file is not None:
            try:
                csv_file.seek(0)
                try:
                    df_input = pd.read_csv(csv_file)
                except UnicodeDecodeError:
                    csv_file.seek(0)
                    df_input = pd.read_csv(csv_file, encoding="latin-1")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return

            cols = list(df_input.columns)
            csv_id_col = st.session_state.get("csv_id_col", cols[0])
            csv_text_cols = st.session_state.get("csv_text_cols", [c for c in cols if "Question" in c or df_input[c].dtype == object])
            if not csv_text_cols:
                csv_text_cols = cols

            total = len(df_input)
            for i, (_, row_data) in enumerate(df_input.iterrows(), start=1):
                filename = str(row_data.get(csv_id_col, f"row_{i}"))
                status.text(f"Marking {filename} ({i}/{total}) from CSV...")

                # Build script text by concatenating selected columns
                parts = []
                for col in csv_text_cols:
                    val = row_data.get(col, "")
                    if isinstance(val, float) and pd.isna(val):
                        continue
                    parts.append(f"{col}:
{val}")
                script_text = "

".join(parts).strip()

                row = {"filename": filename}

                for model_name, label in selected_models:
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
                progress.progress(i / total)

        # === Branch 2: PDF/DOCX scripts ===
        elif script_files:
            total = len(script_files)
            for i, uploaded in enumerate(script_files, start=1):
                filename = uploaded.name
                status.text(f"Marking {filename} ({i}/{total})...")

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

                for model_name, label in selected_models:
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
                progress.progress(i / total)

        if not rows:
            st.warning("No results to show.")
            sidebar_status.info("📚 Ready to mark")
            return

        df = pd.DataFrame(rows)

        # Add comparison columns dynamically based on available *_mark columns
        mark_cols = [c for c in df.columns if c.endswith("_mark")]

        if len(mark_cols) >= 2:
            # Average of all selected models
            df["average_mark"] = df[mark_cols].mean(axis=1)

            # If exactly two models, provide a simple difference column as well
            if len(mark_cols) == 2:
                df["mark_difference"] = (df[mark_cols[0]] - df[mark_cols[1]]).abs()

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

    main()
