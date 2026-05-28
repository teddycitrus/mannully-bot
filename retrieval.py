"""
retrieval.py
============
Hybrid retrieval: BM25 (FTS5) + Gemini vector similarity, fused with
Reciprocal Rank Fusion.

Why hybrid
----------
BM25 nails proper-noun / in-joke recall (Discord's bread and butter) but
misses paraphrases ("upset" vs "pissed off"). Embeddings catch the
paraphrases but blur on names. Fusing top-K from each gives both.

Architecture
------------
* Per-guild in-memory cache of (matrix, rowid_pks). Built lazily on the
  first vector query for a guild; appended to whenever a new message is
  embedded live. Vectors are L2-normalised at insert time so cosine
  collapses to a single matrix-vector dot product.
* Cache lives for the process lifetime - this bot is a single 24/7
  container, so "load once, append forever" is the cheapest path. RAM
  cost is ~3 KB per archived message per guild. At ~100k messages that's
  ~300 MB, which is the realistic ceiling for this design. If a guild
  exceeds that we'd swap to Turso's native vector_distance_cos instead.

All functions here are synchronous and the locks are non-reentrant. Async
callers wrap them in asyncio.to_thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np

import database
import embeddings

log = logging.getLogger("retrieval")

# Reciprocal Rank Fusion constant. 60 is the value used in the original
# Cormack et al. paper and is what every production hybrid retriever
# uses; the precise value barely affects ranking.
_RRF_K = 60

# Per-guild cache: guild_id (str or "" for global) -> (matrix, rowid_pks).
# matrix is float32 of shape (N, EMBEDDING_DIM), rows L2-normalised.
_CACHE: dict[str, tuple[np.ndarray, list[int]]] = {}
_CACHE_LOCK = threading.RLock()


def _cache_key(guild_id: Optional[str]) -> str:
    return str(guild_id) if guild_id is not None else ""


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def _bytes_to_vec(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.shape != (embeddings.EMBEDDING_DIM,):
        raise ValueError(
            f"stored embedding has wrong shape {arr.shape}; "
            f"want ({embeddings.EMBEDDING_DIM},)"
        )
    # frombuffer gives a read-only view; copy so the normalisation can write.
    return _l2_normalize(arr.copy())


def _build_cache(guild_id: Optional[str]) -> tuple[np.ndarray, list[int]]:
    """Load every embedding for a guild from the DB into a normalised matrix."""
    rows = database.load_guild_embedding_rows(guild_id)
    if not rows:
        return (
            np.zeros((0, embeddings.EMBEDDING_DIM), dtype=np.float32),
            [],
        )
    rowid_pks: list[int] = []
    vecs: list[np.ndarray] = []
    for r in rows:
        try:
            vecs.append(_bytes_to_vec(r["embedding"]))
            rowid_pks.append(int(r["rowid_pk"]))
        except ValueError:
            # Skip malformed rows rather than poison the whole cache.
            log.warning("dropping malformed embedding rowid_pk=%s", r.get("rowid_pk"))
    matrix = np.vstack(vecs) if vecs else np.zeros(
        (0, embeddings.EMBEDDING_DIM), dtype=np.float32
    )
    return matrix, rowid_pks


def _get_cache(guild_id: Optional[str]) -> tuple[np.ndarray, list[int]]:
    key = _cache_key(guild_id)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit
        log.info("building vector cache for guild=%s", key or "(global)")
        built = _build_cache(guild_id)
        _CACHE[key] = built
        log.info(
            "vector cache built: %d vectors, %.1f MB",
            built[0].shape[0],
            built[0].nbytes / 1024 / 1024,
        )
        return built


def append_to_cache(
    guild_id: Optional[str], rowid_pk: int, embedding: np.ndarray
) -> None:
    """Append one newly-embedded message to the cache for its guild.

    Skipped silently if the cache hasn't been built yet for that guild;
    the next query will build it from scratch and pick this row up.
    """
    key = _cache_key(guild_id)
    with _CACHE_LOCK:
        cur = _CACHE.get(key)
        if cur is None:
            return
        matrix, rowid_pks = cur
        normed = _l2_normalize(embedding.astype(np.float32, copy=False))
        new_matrix = np.vstack([matrix, normed[None, :]])
        rowid_pks = rowid_pks + [int(rowid_pk)]
        _CACHE[key] = (new_matrix, rowid_pks)


def invalidate_cache(guild_id: Optional[str] = None) -> None:
    """Drop one guild's cache (or all caches if guild_id is None)."""
    with _CACHE_LOCK:
        if guild_id is None:
            _CACHE.clear()
        else:
            _CACHE.pop(_cache_key(guild_id), None)


def vector_search(
    question: str,
    limit: int,
    guild_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Top-`limit` messages by Gemini embedding cosine similarity.

    Returns rows in the same shape as database.search_by_keyword. Empty
    list if the guild has no embeddings yet (backfill hasn't reached
    it).
    """
    if not question.strip() or limit <= 0:
        return []
    matrix, rowid_pks = _get_cache(guild_id)
    if matrix.shape[0] == 0:
        return []
    q = _l2_normalize(embeddings.embed_query(question))
    # Matrix is row-normalised, q is normalised -> dot product == cosine.
    scores = matrix @ q
    k = min(limit, scores.shape[0])
    # argpartition is O(N), faster than a full sort when N >> K.
    if k < scores.shape[0]:
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
    else:
        idx = np.argsort(-scores)
    picked = [rowid_pks[i] for i in idx]
    return database.messages_by_rowids(picked)


def _rrf_fuse(
    rankings: list[list[dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion. Each ranking is in best-first order.

    Documents are keyed by message_id (stable, unique). RRF score is
    sum(1 / (k + rank)) across rankings the doc appears in.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            mid = str(doc.get("message_id"))
            if not mid:
                continue
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            # First sighting wins for the canonical row (they're identical
            # anyway because both rankings hit the same `messages` table).
            by_id.setdefault(mid, doc)
    ordered_ids = sorted(scores.keys(), key=lambda m: -scores[m])
    return [by_id[m] for m in ordered_ids[:limit]]


def hybrid_search(
    question: str,
    limit: int,
    guild_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """BM25 + vector top-K fused via RRF. Falls back gracefully:

    * If the guild has no embeddings yet, returns pure BM25.
    * If embedding the query fails for any reason, returns pure BM25.
    """
    bm25_hits = database.search_by_keyword(question, limit, guild_id)
    try:
        vec_hits = vector_search(question, limit, guild_id)
    except Exception:
        log.exception("vector search failed; falling back to BM25 only")
        vec_hits = []
    if not vec_hits:
        return bm25_hits[:limit]
    return _rrf_fuse([bm25_hits, vec_hits], limit)
