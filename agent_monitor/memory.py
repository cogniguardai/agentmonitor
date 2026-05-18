"""
agent_monitor.memory — persistent text memory with optional semantic search.

Two storage modes:
  1. Plain text (always works): just stores `text` in memory_chunk.
  2. Vector (when interp_bridge has nomic-embed-text reachable):
     also stores a float32 embedding blob and embed_dim. search_semantic()
     then ranks chunks by cosine similarity.

Embedding dim for nomic-embed-text is 768 (~3 KB per chunk on disk).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from agent_monitor import db
from agent_monitor.interp_bridge import _load_once


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32, copy=False).tobytes()


def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def _embed(text: str) -> Optional[np.ndarray]:
    """Try to embed via the interp bridge's nomic-embed-text wrapper.
    Returns None if Ollama is unreachable or the embed model isn't pulled."""
    b = _load_once()
    if b.embed_one is None:
        return None
    try:
        return b.embed_one(text[:6000].strip()).astype(np.float32, copy=False)
    except Exception:
        return None


def remember(
    text: str, *, source: str = "manual", kind: str = "note",
    tags: Iterable[str] = (), with_vector: bool = True,
) -> int:
    """Store a chunk. Returns its row id."""
    if not text or not text.strip():
        raise ValueError("memory.remember: empty text")
    text = text.strip()

    blob = None
    dim = None
    if with_vector:
        v = _embed(text)
        if v is not None:
            blob = _vec_to_blob(v)
            dim = int(v.shape[0])

    with db.session() as conn:
        return db.add_memory(
            conn, source=source, kind=kind, text=text,
            embedding=blob, embed_dim=dim, tags=tags,
        )


def search_text(query: str, *, limit: int = 20) -> List[dict]:
    """Plain SQL LIKE search. No embeddings needed."""
    q = f"%{query.strip()}%"
    with db.session() as conn:
        rows = conn.execute(
            "SELECT id, source, kind, text, tags, created_at "
            "FROM memory_chunk WHERE text LIKE ? "
            "ORDER BY id DESC LIMIT ?",
            (q, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def search_semantic(query: str, *, limit: int = 10) -> List[Tuple[dict, float]]:
    """Cosine-similarity search over chunks that have an embedding stored.
    Returns [(chunk_dict, score), ...] sorted by score desc."""
    qv = _embed(query)
    if qv is None:
        return []
    qn = qv / (np.linalg.norm(qv) + 1e-9)

    out: List[Tuple[dict, float]] = []
    with db.session() as conn:
        rows = conn.execute(
            "SELECT id, source, kind, text, tags, embedding_blob, embed_dim, created_at "
            "FROM memory_chunk WHERE embedding_blob IS NOT NULL"
        ).fetchall()
    for r in rows:
        v = _blob_to_vec(r["embedding_blob"], r["embed_dim"])
        score = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-9)))
        d = dict(r)
        d.pop("embedding_blob", None)
        out.append((d, score))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:limit]


def list_recent(limit: int = 50) -> List[dict]:
    with db.session() as conn:
        rows = conn.execute(
            "SELECT id, source, kind, text, tags, embed_dim, created_at "
            "FROM memory_chunk ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
