import json
import re
import shutil
from pathlib import Path
from typing import Any

from .core import ROOT_DIR


CACHE_ROOT = ROOT_DIR / ".scriptgrade_cache"
LAST_INGEST_DIR = CACHE_ROOT / "last_ingest"
LAST_INGEST_MANIFEST = LAST_INGEST_DIR / "manifest.json"


def save_ingest_snapshot(
    *,
    use_assessment_folders: bool,
    single_assessment_folder_mode: bool,
    assessment_root: str,
    folder_keywords: dict[str, tuple[str, ...]],
    scale_profile: str,
    manual_max_mark: float | None,
    document_only_mode: bool,
    script_files: list[Any] | None,
    csv_file: Any | None,
    rubric_files: list[Any] | None,
    brief_files: list[Any] | None,
    marking_scheme_files: list[Any] | None,
    graded_sample_files: list[Any] | None,
    other_files: list[Any] | None,
) -> Path:
    if LAST_INGEST_DIR.exists():
        shutil.rmtree(LAST_INGEST_DIR)
    LAST_INGEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "use_assessment_folders": use_assessment_folders,
        "single_assessment_folder_mode": single_assessment_folder_mode,
        "assessment_root": assessment_root.strip(),
        "folder_keywords": {key: list(value) for key, value in folder_keywords.items()},
        "scale_profile": scale_profile,
        "manual_max_mark": manual_max_mark,
        "document_only_mode": document_only_mode,
        "script_files": _save_uploaded_files(script_files, LAST_INGEST_DIR / "scripts"),
        "csv_file": _save_uploaded_file(csv_file, LAST_INGEST_DIR / "csv"),
        "documents": {
            "rubric_files": _save_uploaded_files(rubric_files, LAST_INGEST_DIR / "rubric"),
            "brief_files": _save_uploaded_files(brief_files, LAST_INGEST_DIR / "brief"),
            "marking_scheme_files": _save_uploaded_files(marking_scheme_files, LAST_INGEST_DIR / "marking_scheme"),
            "graded_sample_files": _save_uploaded_files(graded_sample_files, LAST_INGEST_DIR / "graded_sample"),
            "other_files": _save_uploaded_files(other_files, LAST_INGEST_DIR / "other"),
        },
    }

    LAST_INGEST_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return LAST_INGEST_MANIFEST


def load_ingest_snapshot() -> dict[str, Any] | None:
    if not LAST_INGEST_MANIFEST.exists():
        return None
    return json.loads(LAST_INGEST_MANIFEST.read_text(encoding="utf-8"))


def get_last_ingest_manifest_path() -> Path:
    return LAST_INGEST_MANIFEST


def _save_uploaded_files(uploaded_files: list[Any] | None, target_dir: Path) -> list[str]:
    if not uploaded_files:
        return []
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        filename = f"{index:03d}_{_sanitize_filename(uploaded_file.name)}"
        destination = target_dir / filename
        destination.write_bytes(uploaded_file.getvalue())
        saved_paths.append(str(destination))
    return saved_paths


def _save_uploaded_file(uploaded_file: Any | None, target_dir: Path) -> str:
    if uploaded_file is None:
        return ""
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / _sanitize_filename(uploaded_file.name)
    destination.write_bytes(uploaded_file.getvalue())
    return str(destination)


def _sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip())
    return cleaned or "file"
