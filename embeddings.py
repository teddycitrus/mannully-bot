"""
embeddings.py
=============
Thin wrapper around Gemini's text-embedding-004 for the hybrid retriever.

Two call shapes
---------------
* embed_documents(texts) -> list[np.ndarray] for archive content (task_type=
  RETRIEVAL_DOCUMENT).
* embed_query(text) -> np.ndarray for a live question (task_type=
  RETRIEVAL_QUERY).

The distinction matters: Gemini produces asymmetric vectors tuned for one
side or the other, and mixing them up degrades recall.

All vectors are float32 numpy arrays of length EMBEDDING_DIM. Bytes on
disk are the raw float32 buffer (np.ndarray.tobytes()), so a stored
embedding is exactly EMBEDDING_DIM * 4 bytes (3072 for 768-dim).

This module is intentionally sync. Callers wrap it in asyncio.to_thread.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import google.generativeai as genai

import config

log = logging.getLogger("embeddings")

# gemini-embedding-001 is the current stable Gemini embedding model. Its
# native dimensionality is 3072; we truncate to 768 via output_dimensionality
# to keep on-disk and in-RAM cost at ~3 KB per message. Google's docs note
# that any truncated output should be L2-normalised, which retrieval.py
# already does on cache load.
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768
EMBEDDING_BYTES = EMBEDDING_DIM * 4  # float32

# Gemini batch endpoint accepts a list under "content". Keep batches modest
# so a single bad input doesn't waste a big API call's worth of work.
MAX_BATCH = 100

# Crude exponential backoff for transient errors (429s, brief 5xx).
_MAX_RETRIES = 4
_BASE_BACKOFF = 1.5


def _to_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"unexpected embedding shape {arr.shape}, want ({EMBEDDING_DIM},)"
        )
    return arr


def _embed(texts: list[str], task_type: str) -> list[np.ndarray]:
    """Call Gemini's embed_content with retry. Returns one vector per input."""
    if not texts:
        return []
    attempt = 0
    while True:
        try:
            # The SDK accepts either a single string or a list; with a list
            # it returns {"embedding": [[...], [...], ...]} in input order.
            resp = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            )
            raw = resp["embedding"] if isinstance(resp, dict) else resp.embedding
            if len(texts) == 1 and raw and isinstance(raw[0], (int, float)):
                # Single-input call sometimes returns a flat list; wrap it.
                raw = [raw]
            if len(raw) != len(texts):
                raise RuntimeError(
                    f"embedding count mismatch: got {len(raw)}, want {len(texts)}"
                )
            return [_to_array(v) for v in raw]
        except Exception as exc:
            attempt += 1
            if attempt > _MAX_RETRIES:
                log.error(
                    "embedding call failed after %d retries: %s", _MAX_RETRIES, exc
                )
                raise
            backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
            log.warning(
                "embedding call transient error (%s); retry %d/%d in %.1fs",
                exc, attempt, _MAX_RETRIES, backoff,
            )
            time.sleep(backoff)


def embed_documents(texts: Sequence[str]) -> list[np.ndarray]:
    """Embed archive content. Splits into MAX_BATCH-sized API calls."""
    out: list[np.ndarray] = []
    batch: list[str] = []
    for t in texts:
        batch.append(t if t else " ")  # Gemini rejects empty strings
        if len(batch) >= MAX_BATCH:
            out.extend(_embed(batch, task_type="RETRIEVAL_DOCUMENT"))
            batch = []
    if batch:
        out.extend(_embed(batch, task_type="RETRIEVAL_DOCUMENT"))
    return out


def embed_query(text: str) -> np.ndarray:
    """Embed a live user question."""
    return _embed([text if text else " "], task_type="RETRIEVAL_QUERY")[0]


# Ensure the SDK is configured before the first call. config.py already
# configures it inside gemini_client.py, but importing this module without
# gemini_client first should still work.
genai.configure(api_key=config.GEMINI_API_KEY)
