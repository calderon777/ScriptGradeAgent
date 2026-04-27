"""
Audit logger — singleton-style append-only log of all LLM calls.

Usage:
    from experiments.cross_sectional_part_marker.src.audit import AuditLog

    log = AuditLog.instance()
    log.append_entry(entry)
    log.save(Path("outputs/audit_trail.jsonl"))
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from threading import Lock
from typing import ClassVar

from experiments.cross_sectional_part_marker.src.schemas import AuditEntry

logger = logging.getLogger(__name__)


class AuditLog:
    """Thread-safe singleton audit log for all pipeline LLM calls."""

    _instance: ClassVar[AuditLog | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._entry_lock = Lock()

    @classmethod
    def instance(cls) -> "AuditLog":
        """Return the global singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful in tests)."""
        with cls._lock:
            cls._instance = None

    def append_entry(self, entry: AuditEntry) -> None:
        """Append an audit entry thread-safely."""
        with self._entry_lock:
            self._entries.append(entry)
            logger.debug(
                "Audit entry appended: stage=%s model=%s submission=%s",
                entry.stage,
                entry.model_name,
                entry.submission_id,
            )

    def get_entries(self) -> list[AuditEntry]:
        """Return a snapshot of all current entries."""
        with self._entry_lock:
            return list(self._entries)

    def save(self, path: Path) -> None:
        """Write all entries to a JSONL file (overwrites)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._entry_lock:
            entries = list(self._entries)
        with open(path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(entry.model_dump_json() + "\n")
        logger.info("Audit log saved to %s (%d entries)", path, len(entries))

    def load(self, path: Path) -> None:
        """Load entries from an existing JSONL file (appends to current log)."""
        path = Path(path)
        if not path.exists():
            logger.warning("Audit log file not found: %s", path)
            return
        loaded = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = AuditEntry.model_validate_json(line)
                    self.append_entry(entry)
                    loaded += 1
                except Exception as exc:
                    logger.warning("Failed to parse audit line: %s — %s", line[:80], exc)
        logger.info("Loaded %d audit entries from %s", loaded, path)

    @staticmethod
    def get_prompt_hash(prompt: str) -> str:
        """Return the SHA-256 hex digest of a prompt string."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def get_response_hash(response: str) -> str:
        """Return the SHA-256 hex digest of a response string."""
        return hashlib.sha256(response.encode("utf-8")).hexdigest()
