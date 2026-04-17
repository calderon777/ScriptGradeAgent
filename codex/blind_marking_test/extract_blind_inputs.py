from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs).strip()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ec3040_marking_scheme.txt": extract_pdf_text(INPUT_DIR / "ec3040_marking_scheme.pdf"),
        "alessandro_submission.txt": extract_docx_text(INPUT_DIR / "alessandro_submission.docx"),
        "fahima_submission.txt": extract_pdf_text(INPUT_DIR / "fahima_submission.pdf"),
    }
    for name, text in outputs.items():
        (OUTPUT_DIR / name).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
