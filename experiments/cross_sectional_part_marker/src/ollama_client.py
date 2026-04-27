"""
Abstraction layer for all Ollama model calls (generate + embed).

All LLM communication in this pipeline goes through OllamaClient.
No data is ever sent to an external API.

Usage:
    client = OllamaClient(base_url="http://localhost:11434", timeout=120)
    text = client.generate("qwen2.5:7b", "Explain supply and demand")
    vec  = client.embed("nomic-embed-text", "Supply shifts right when cost falls")
    entries = client.get_audit_entries()
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import requests

from experiments.cross_sectional_part_marker.src.audit import AuditLog
from experiments.cross_sectional_part_marker.src.schemas import AuditEntry

logger = logging.getLogger(__name__)

_RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds — exponential back-off for 3 retries


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama is not reachable after all retries."""


class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.

    All calls are local (http://localhost:...). Retries on connection errors
    with exponential back-off. Every call is recorded in the shared AuditLog.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        stage: str = "unknown",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._stage = stage
        self._audit = AuditLog.instance()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """
        Call Ollama /api/generate and return the response text.

        Parameters
        ----------
        model:       Ollama model tag e.g. "qwen2.5:7b"
        prompt:      The user prompt
        system:      Optional system prompt
        temperature: Sampling temperature (0 = deterministic)

        Returns
        -------
        str — the model's response text

        Raises
        ------
        OllamaConnectionError if Ollama is unreachable after 3 retries
        """
        url = f"{self._base_url}/api/generate"
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        prompt_hash = AuditLog.get_prompt_hash(prompt)
        t0 = time.perf_counter()
        raw = self._post_with_retry(url, payload)
        elapsed = time.perf_counter() - t0

        response_text: str = raw.get("response", "")
        logger.info(
            "generate | model=%s | elapsed=%.2fs | prompt_chars=%d | response_chars=%d",
            model,
            elapsed,
            len(prompt),
            len(response_text),
        )

        self._record_audit(
            model=model,
            prompt_hash=prompt_hash,
            prompt=prompt,
            response=response_text,
            notes=f"elapsed={elapsed:.2f}s",
        )
        return response_text

    def embed(self, model: str, text: str) -> list[float]:
        """
        Call Ollama /api/embeddings and return the embedding vector.

        Parameters
        ----------
        model: Ollama embedding model tag e.g. "nomic-embed-text"
        text:  The text to embed

        Returns
        -------
        list[float] — embedding vector

        Raises
        ------
        OllamaConnectionError if Ollama is unreachable after 3 retries
        """
        url = f"{self._base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}

        prompt_hash = AuditLog.get_prompt_hash(text)
        t0 = time.perf_counter()
        raw = self._post_with_retry(url, payload)
        elapsed = time.perf_counter() - t0

        embedding: list[float] = raw.get("embedding", [])
        logger.info(
            "embed | model=%s | elapsed=%.2fs | text_chars=%d | dims=%d",
            model,
            elapsed,
            len(text),
            len(embedding),
        )

        self._record_audit(
            model=model,
            prompt_hash=prompt_hash,
            prompt=text,
            response=str(len(embedding)),
            notes=f"embed dims={len(embedding)} elapsed={elapsed:.2f}s",
        )
        return embedding

    def get_audit_entries(self) -> list[AuditEntry]:
        """Return all audit entries recorded by this client."""
        return self._audit.get_entries()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_with_retry(self, url: str, payload: dict) -> dict:
        """POST payload to url, retrying up to 3 times on connection errors."""
        last_exc: Exception | None = None
        for attempt, delay in enumerate([0.0] + list(_RETRY_DELAYS)):
            if delay:
                logger.warning("Retrying in %.1fs (attempt %d)…", delay, attempt)
                time.sleep(delay)
            try:
                resp = requests.post(url, json=payload, timeout=self._timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                logger.warning("Connection error to %s: %s", url, exc)
            except requests.exceptions.Timeout as exc:
                last_exc = exc
                logger.warning("Timeout calling %s after %ds: %s", url, self._timeout, exc)
            except requests.exceptions.HTTPError as exc:
                # Don't retry HTTP errors (4xx/5xx are not transient)
                raise OllamaConnectionError(f"HTTP error from Ollama: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise OllamaConnectionError(f"Invalid JSON from Ollama: {exc}") from exc

        raise OllamaConnectionError(
            f"Ollama not reachable at {url} after {len(_RETRY_DELAYS) + 1} attempts. "
            "Ensure `ollama serve` is running."
        ) from last_exc

    def _record_audit(
        self,
        model: str,
        prompt_hash: str,
        prompt: str = "",
        response: str = "",
        notes: str = "",
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=self._stage,
            model_name=model,
            prompt_hash=prompt_hash,
            response_hash=AuditLog.get_response_hash(response),
            prompt_text=prompt,
            response_text=response,
            notes=notes,
        )
        self._audit.append_entry(entry)
