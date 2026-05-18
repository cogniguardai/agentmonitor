"""
agent_monitor.nla_cache -- content-addressed cache for NLA decodings.

Why this exists
---------------
Decoding the same text twice is wasteful and, for prompted_approx, expensive
(30-60s per call on CPU). For commercial deployments where many runs share
boilerplate (system prompts, refusals, common customer-support phrasings),
the same text gets re-decoded thousands of times across the lifetime of a
deployment.

This module provides a small, content-addressed cache:
    key = sha256(backend || model || text)
    value = the decoding dict
    ttl   = 24 hours by default

Two layers:
    L1: in-process OrderedDict LRU (bounded, fast, lost on restart)
    L2: SQLite table `nla_cache` (durable, persists across restarts)

Both layers are thread-safe. The worker thread writes through L1 to L2.
Reads check L1 first, then L2 (and warm L1 on L2 hit).

Cache keys are content-addressed and never include user PII as a key, so
collisions across users are intentional and safe (same input -> same
explanation, by design). If a deployment cannot tolerate cross-user cache
sharing it can disable the cache entirely with NLA_CACHE_DISABLED=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from agent_monitor import db

DEFAULT_TTL_S = int(os.environ.get("NLA_CACHE_TTL", "86400"))   # 24 h
L1_MAX = int(os.environ.get("NLA_CACHE_L1_MAX", "256"))
DISABLED = os.environ.get("NLA_CACHE_DISABLED", "0") == "1"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nla_cache (
    key         TEXT PRIMARY KEY,
    backend     TEXT NOT NULL,
    model       TEXT,
    decoding_json TEXT NOT NULL,
    created_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS nla_cache_created_idx ON nla_cache(created_at);
"""


_lock = threading.RLock()
_l1: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0, "writes": 0, "evictions": 0}
_initialised = False


def _ensure_schema() -> None:
    global _initialised
    if _initialised:
        return
    try:
        with db.session() as conn:
            conn.executescript(_SCHEMA)
        _initialised = True
    except Exception:
        # If the DB is unreachable we silently degrade to L1-only.
        _initialised = True


def make_key(backend: str, model: Optional[str], text: str) -> str:
    h = hashlib.sha256()
    h.update((backend or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((model or "").encode("utf-8"))
    h.update(b"\x1f")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()


def _l1_get(key: str) -> Optional[dict]:
    item = _l1.get(key)
    if item is None:
        return None
    ts, val = item
    if time.time() - ts > DEFAULT_TTL_S:
        _l1.pop(key, None)
        return None
    # promote to MRU
    _l1.move_to_end(key)
    return val


def _l1_put(key: str, value: dict) -> None:
    _l1[key] = (time.time(), value)
    _l1.move_to_end(key)
    while len(_l1) > L1_MAX:
        _l1.popitem(last=False)
        _stats["evictions"] += 1


def get(backend: str, model: Optional[str], text: str) -> Optional[dict]:
    if DISABLED or not text:
        return None
    _ensure_schema()
    key = make_key(backend, model, text)
    with _lock:
        v = _l1_get(key)
        if v is not None:
            _stats["l1_hits"] += 1
            return _clone(v)
    # L2
    try:
        with db.session() as conn:
            cur = conn.execute(
                "SELECT decoding_json, created_at FROM nla_cache WHERE key = ?",
                (key,),
            )
            row = cur.fetchone()
            if not row:
                with _lock:
                    _stats["misses"] += 1
                return None
            if time.time() - float(row["created_at"]) > DEFAULT_TTL_S:
                conn.execute("DELETE FROM nla_cache WHERE key = ?", (key,))
                with _lock:
                    _stats["misses"] += 1
                return None
            conn.execute(
                "UPDATE nla_cache SET hits = hits + 1 WHERE key = ?", (key,)
            )
            decoding = json.loads(row["decoding_json"])
    except Exception:
        with _lock:
            _stats["misses"] += 1
        return None
    with _lock:
        _l1_put(key, decoding)
        _stats["l2_hits"] += 1
    return _clone(decoding)


def put(backend: str, model: Optional[str], text: str, decoding: dict) -> None:
    if DISABLED or not text or not decoding or not decoding.get("ok"):
        return
    _ensure_schema()
    key = make_key(backend, model, text)
    blob = json.dumps(decoding, default=str)
    with _lock:
        _l1_put(key, decoding)
        _stats["writes"] += 1
    try:
        with db.session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nla_cache "
                "(key, backend, model, decoding_json, created_at, hits) "
                "VALUES (?, ?, ?, ?, ?, COALESCE("
                "  (SELECT hits FROM nla_cache WHERE key = ?), 0))",
                (key, backend, model, blob, time.time(), key),
            )
    except Exception:
        pass


def stats() -> Dict[str, Any]:
    _ensure_schema()
    out = {**_stats, "l1_size": len(_l1), "l2_size": 0, "disabled": DISABLED}
    try:
        with db.session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(hits), 0) AS h FROM nla_cache"
            ).fetchone()
            if row:
                out["l2_size"] = int(row["n"])
                out["l2_total_hits"] = int(row["h"])
    except Exception:
        pass
    total = out["l1_hits"] + out["l2_hits"] + out["misses"]
    out["hit_rate"] = (
        (out["l1_hits"] + out["l2_hits"]) / total if total else None
    )
    return out


def clear() -> Dict[str, Any]:
    _ensure_schema()
    cleared_l2 = 0
    with _lock:
        _l1.clear()
    try:
        with db.session() as conn:
            cur = conn.execute("DELETE FROM nla_cache")
            cleared_l2 = cur.rowcount or 0
    except Exception:
        pass
    return {"cleared_l1": True, "cleared_l2": cleared_l2}


def _clone(d: dict) -> dict:
    """Return a shallow copy with a 'cached': True marker so callers can tell."""
    out = dict(d)
    out["cached"] = True
    return out
