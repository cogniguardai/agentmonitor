"""
agent_monitor.nla_worker -- bounded async background decoder for NLA.

Why this exists
---------------
Decoding takes 30-90 seconds per call on `prompted_approx` (CPU Ollama). We
don't want to block agents on that. This module owns a single worker thread
that drains a bounded queue of decode jobs, calls nla_client.decode(), and
persists the result via db.record_nla_decoding().

Design
------
- One worker thread (more would oversubscribe a single Ollama instance and
  thrash the model in/out of RAM).
- Bounded queue (default 64). Drop-OLDEST policy when full -- recent agent
  output is more interesting than ancient backlog.
- The worker is started lazily on first enqueue, joined cleanly on shutdown.
- Every job is keyed by (run_id, target, text-hash) so re-enqueueing the same
  text is a no-op when there is already a pending job for it. Also: the cache
  in nla_cache.get() short-circuits before any expensive work runs.
- All state is observable via stats(): queue depth, processed, dropped,
  duplicates, errors, current job (or None).

The worker NEVER raises into callers. enqueue_decode() returns a status string
and silently drops on backpressure.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

QUEUE_MAX = int(os.environ.get("NLA_QUEUE_MAX", "64"))
WORKER_ENABLED = os.environ.get("NLA_WORKER_ENABLED", "1") == "1"


@dataclass
class _Job:
    run_id: Optional[int]
    target: str                   # 'input' | 'output' | 'cot' | 'adhoc'
    text: str
    trace_seq: Optional[int] = None
    enqueued_at: float = field(default_factory=time.time)
    key: str = ""

    def short(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target": self.target,
            "trace_seq": self.trace_seq,
            "text_chars": len(self.text or ""),
            "enqueued_at": self.enqueued_at,
            "age_s": round(time.time() - self.enqueued_at, 2),
        }


class _NLAWorker:
    def __init__(self, max_queue: int = QUEUE_MAX):
        self._queue: "deque[_Job]" = deque()
        self._pending_keys: "OrderedDict[str, _Job]" = OrderedDict()
        self._max_queue = max_queue
        self._cv = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._current: Optional[_Job] = None
        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "dropped_full": 0,
            "deduped": 0,
            "errors": 0,
            "last_processed_at": None,
            "last_error": None,
        }
        self._lat: "deque[int]" = deque(maxlen=100)

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        with self._cv:
            if self._thread and self._thread.is_alive():
                return
            self._shutdown = False
            t = threading.Thread(
                target=self._run, name="nla-worker", daemon=True
            )
            self._thread = t
            t.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)

    # -- enqueue / dedupe ----------------------------------------------

    @staticmethod
    def _key(run_id: Optional[int], target: str, text: str) -> str:
        h = hashlib.sha256()
        h.update(str(run_id or "").encode())
        h.update(b"\x1f")
        h.update((target or "").encode())
        h.update(b"\x1f")
        h.update((text or "").encode("utf-8"))
        return h.hexdigest()

    def enqueue(
        self, *, run_id: Optional[int], target: str, text: str,
        trace_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not WORKER_ENABLED:
            return {"queued": False, "reason": "worker disabled"}
        if not text or not text.strip():
            return {"queued": False, "reason": "empty text"}
        key = self._key(run_id, target, text)
        job = _Job(
            run_id=run_id, target=target, text=text,
            trace_seq=trace_seq, key=key,
        )
        with self._cv:
            if key in self._pending_keys:
                self._stats["deduped"] += 1
                return {"queued": False, "reason": "duplicate", "key": key}
            # backpressure: drop oldest when full
            while len(self._queue) >= self._max_queue:
                old = self._queue.popleft()
                self._pending_keys.pop(old.key, None)
                self._stats["dropped_full"] += 1
            self._queue.append(job)
            self._pending_keys[key] = job
            self._stats["enqueued"] += 1
            self._cv.notify()
        # Lazy-start the worker on first enqueue
        if not (self._thread and self._thread.is_alive()):
            self.start()
        return {"queued": True, "key": key, "depth": len(self._queue)}

    # -- observability -------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._cv:
            depth = len(self._queue)
            cur = self._current.short() if self._current else None
            lat = list(self._lat)
            stats = dict(self._stats)
        avg = (sum(lat) / len(lat)) if lat else None
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "enabled": WORKER_ENABLED,
            "queue_depth": depth,
            "queue_max": self._max_queue,
            "current": cur,
            "avg_latency_ms": int(avg) if avg is not None else None,
            **stats,
        }

    # -- the actual loop ----------------------------------------------

    def _run(self) -> None:
        # Imported lazily to avoid circular imports at module load.
        from agent_monitor import db, nla_client

        while True:
            with self._cv:
                while not self._queue and not self._shutdown:
                    self._cv.wait(timeout=2.0)
                if self._shutdown and not self._queue:
                    return
                job = self._queue.popleft()
                self._pending_keys.pop(job.key, None)
                self._current = job
            t0 = time.time()
            try:
                result = nla_client.decode(job.text)
            except Exception as e:
                with self._cv:
                    self._stats["errors"] += 1
                    self._stats["last_error"] = (
                        f"{type(e).__name__}: {e}"
                    )
                    self._current = None
                continue
            elapsed_ms = int((time.time() - t0) * 1000)
            with self._cv:
                self._lat.append(elapsed_ms)
                self._stats["processed"] += 1
                self._stats["last_processed_at"] = time.time()
                self._current = None
            # Persist to DB. Soft-fail.
            if result.get("ok"):
                try:
                    with db.session() as conn:
                        db.record_nla_decoding(
                            conn,
                            run_id=job.run_id,
                            target=job.target,
                            decoding=result,
                            trace_seq=job.trace_seq,
                        )
                except Exception as e:
                    with self._cv:
                        self._stats["errors"] += 1
                        self._stats["last_error"] = (
                            f"persist failed: {type(e).__name__}: {e}"
                        )


# -- module-level singleton ------------------------------------------

_worker: Optional[_NLAWorker] = None
_singleton_lock = threading.Lock()


def get_worker() -> _NLAWorker:
    global _worker
    if _worker is None:
        with _singleton_lock:
            if _worker is None:
                _worker = _NLAWorker(max_queue=QUEUE_MAX)
    return _worker


def enqueue_decode(
    *, run_id: Optional[int], target: str, text: str,
    trace_seq: Optional[int] = None,
) -> Dict[str, Any]:
    """Public API: queue a text for background NLA decoding.

    Returns {"queued": bool, "reason": str?, "key": str?, "depth": int?}.
    Never raises.
    """
    try:
        return get_worker().enqueue(
            run_id=run_id, target=target, text=text, trace_seq=trace_seq,
        )
    except Exception as e:
        return {"queued": False, "reason": f"error: {e}"}


def stats() -> Dict[str, Any]:
    if _worker is None:
        return {
            "running": False,
            "enabled": WORKER_ENABLED,
            "queue_depth": 0,
            "queue_max": QUEUE_MAX,
            "current": None,
            "enqueued": 0, "processed": 0, "dropped_full": 0,
            "deduped": 0, "errors": 0,
        }
    return get_worker().stats()


def shutdown(timeout: float = 5.0) -> None:
    if _worker is not None:
        _worker.stop(timeout=timeout)
