"""
agent_monitor.runner — observe + persist runs of the existing automations.

Design (beginner walkthrough):
    The existing automations (automations/customer_support.py,
    automations/sop_processor.py) already do the real work. We do NOT
    rewrite them. Instead, we wrap their inputs/outputs in a 'MonitoredRun'
    context that:
        1. creates a row in the `run` table
        2. forwards trace events into `trace_event`
        3. scores input + output with the harm/refusal probes
           (interp_bridge) and writes them to `interp_score`
        4. on success, optionally stores the input/output in long-term
           memory (memory_chunk) with embeddings.

This file gives both:
    - run_customer_support_tickets()  -- pre-wired wrapper
    - MonitoredRun                    -- generic context manager for any
                                          new agent we want to register
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from agent_monitor import db

# interp_bridge + memory are optional Phase-2 modules. When missing,
# scoring and memory persistence become no-ops; runs are still recorded
# in the SQLite DB (input/output/trace/elapsed).
try:
    from agent_monitor import interp_bridge
except Exception:
    interp_bridge = None
try:
    from agent_monitor import memory
except Exception:
    memory = None

# NLA_AUTO_DECODE controls async thought-decoding of run input/output:
#   off    -- never queue (default; fully back-compat)
#   output -- queue final agent output for background NLA decoding
#   all    -- queue both input and output
NLA_AUTO_DECODE = os.environ.get("NLA_AUTO_DECODE", "off").lower()
_NLA_AUTO_TARGETS = {
    "off": (),
    "output": ("output",),
    "all": ("input", "output"),
}.get(NLA_AUTO_DECODE, ())


# ---------------------------------------------------------------------------
# Generic context manager -- the building block.
# ---------------------------------------------------------------------------

class MonitoredRun:
    """Single execution; persists input, trace, output, interp scores."""

    def __init__(
        self,
        agent_name: str,
        input_text: str,
        *,
        external_id: Optional[str] = None,
        agent_description: str = "",
        meta: Optional[Dict[str, Any]] = None,
        score_input: bool = True,
    ):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.input_text = input_text
        self.external_id = external_id
        self.meta = dict(meta or {})
        self.score_input = score_input

        self.run_id: Optional[int] = None
        self.agent_id: Optional[int] = None
        self._t0: float = 0.0

    # context manager
    def __enter__(self) -> "MonitoredRun":
        with db.session() as conn:
            self.agent_id = db.upsert_agent(
                conn, self.agent_name, self.agent_description
            )
            self.run_id = db.create_run(
                conn,
                self.agent_id,
                external_id=self.external_id,
                input_text=self.input_text,
                meta=self.meta,
            )
            if self.score_input and self.input_text:
                self._score_and_record(conn, "input", self.input_text)
        self._t0 = time.time()
        if "input" in _NLA_AUTO_TARGETS and self.input_text:
            self._enqueue_nla("input", self.input_text)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = int((time.time() - self._t0) * 1000)
        status = "error" if exc_type else "done"
        with db.session() as conn:
            db.finish_run(
                conn, self.run_id, status=status,
                output_text=getattr(self, "output_text", "") or "",
                elapsed_ms=elapsed_ms,
            )
            if exc_type:
                db.append_trace(conn, self.run_id, "error", {
                    "type": exc_type.__name__,
                    "message": str(exc_val),
                })
        return False  # don't swallow exceptions

    # ---- helpers used by callers ----

    def trace(self, kind: str, payload: Dict[str, Any]) -> None:
        with db.session() as conn:
            db.append_trace(conn, self.run_id, kind, payload)

    def set_output(self, text: str, *, score: bool = True,
                   remember_in_memory: bool = True) -> None:
        self.output_text = text or ""
        if score and self.output_text:
            with db.session() as conn:
                self._score_and_record(conn, "output", self.output_text)
        if "output" in _NLA_AUTO_TARGETS and self.output_text:
            self._enqueue_nla("output", self.output_text)
        if remember_in_memory and self.output_text and memory is not None:
            try:
                memory.remember(
                    self.output_text,
                    source=f"run:{self.run_id}",
                    kind="output",
                    tags=(self.agent_name,),
                )
            except Exception:
                pass  # memory writes never break runs

    def update_meta(self, **kw: Any) -> None:
        self.meta.update(kw)
        with db.session() as conn:
            db.finish_run(conn, self.run_id, status="running", meta=self.meta)

    # ---- internals ----

    def _enqueue_nla(self, target: str, text: str) -> None:
        """Fire-and-forget NLA decoding via the background worker.

        Latency-critical: the worker thread does all the heavy lifting; this
        method just appends a job to the queue and returns immediately.
        """
        try:
            from agent_monitor import nla_worker
            nla_worker.enqueue_decode(
                run_id=self.run_id, target=target, text=text,
            )
        except Exception:
            # NLA must never break a run
            pass

    def _score_and_record(self, conn, target: str, text: str) -> None:
        if interp_bridge is None:
            return  # interp probes not installed; skip scoring
        scores = interp_bridge.score_all(text)
        # Persist only numeric probe scores. `_meta` carries Llama Guard
        # source/category info and goes into the trace event.
        for probe, score in scores.items():
            if probe.startswith("_"):
                continue
            if score is None or not isinstance(score, (int, float)):
                continue
            db.record_interp_score(conn, self.run_id,
                                   target=target, probe=probe, score=float(score))
        # live-feed event includes both the numeric scores and the metadata
        db.append_trace(conn, self.run_id, "interp", {
            "target": target,
            "scores": {k: v for k, v in scores.items() if not k.startswith("_")},
            "meta": scores.get("_meta", {}),
        })


# Note: pre-wired wrappers for Mythos-internal automations
# (run_customer_support_tickets, etc.) used to live here. They were
# tightly coupled to `core.engine`, `core.kairos`, and the `automations/`
# package and have been removed for v0.1.0. The generic MonitoredRun
# context manager above is the public SDK; users build their own wrappers
# around it (or use one of the LLM-specific adapters in
# `agent_monitor.adapters.{openai,anthropic,langchain,...}`).
