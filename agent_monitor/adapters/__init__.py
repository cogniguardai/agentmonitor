"""
agent_monitor.adapters -- universal agent runtime adapters (v1.6).

WHAT THIS IS
============

Until v1.5 AgentMonitor only knew how to ingest traces from our in-house
Qwen/vLLM agents (`customer_support`, `sop_processor`, ...). v1.6 makes
the dashboard work with *any* LLM-agent runtime. The runtime-specific
glue lives in this package: one submodule per adapter
(`ollama.py`, `openai.py`, `anthropic.py`, `langchain.py`, ...).

The contract every adapter must satisfy:

  1. Identify itself with a stable `kind` string
     (e.g. 'ollama', 'openai', 'anthropic'). This goes in `agent.kind`
     so the UI can render correctly and the Interp tab can degrade
     honestly when probes don't exist for the runtime.

  2. Run a turn / a chain / a tool-using loop, and append events to
     the existing `trace_event` table via `append_trace()`. The event
     `kind` field already supports 'model_call' | 'tool' | 'log' which
     covers the vast majority of LLM-agent patterns. Adapters can also
     emit custom kinds; the UI shows them as a generic JSON blob.

  3. On entry, call `db.create_run(...)`; on exit, call
     `db.finish_run(..., status='done'|'error', elapsed_ms=...)`. The
     `MonitoredRun` context manager below does this automatically.

WHAT THIS IS NOT
================

This is *observability*, not orchestration. We do not own the agent
loop. The user's existing chain / agent / runtime keeps running where
it always ran; the adapter is a thin wrapper that copies the events
into our DB. So we never have to keep up with framework churn.

INTERP DEGRADATION
==================

The `interp/` and `interp_real/` probes hook into the residual stream
of a specific transformer architecture (currently Qwen2.5). When a run
came from `kind != 'qwen-vllm'`, the API marks `interp_available=False`
and the UI says so. This is the honesty contract: we never *pretend*
to have signal we don't have.
"""
from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Protocol

from agent_monitor import db


# ---------------------------------------------------------------------------
# Protocol every adapter implements
# ---------------------------------------------------------------------------

class AgentAdapter(Protocol):
    """The minimal contract.

    Concrete adapters subclass / duck-type this. Each adapter typically
    exposes a `run(input_text, *, agent_name, external_id=None, **kwargs)`
    method that internally uses `monitored_run(...)` to handle DB book-
    keeping and trace emission.
    """

    #: Stable identifier for this runtime. Used as `agent.kind`.
    kind: str

    #: Human-readable description for the UI ("OpenAI Chat Completions").
    description: str

    #: Whether AgentMonitor's interp probes can be applied to this runtime.
    #: True only for Qwen-family local models we have probes trained for.
    interp_available: bool

    def run(self, input_text: str, *, agent_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute one turn / chain. Returns {"run_id", "output_text",
        "status", ...}. Implementations must handle their own errors;
        the helper below ensures the DB always reaches a terminal state."""
        ...


# ---------------------------------------------------------------------------
# Helper context manager every adapter uses
# ---------------------------------------------------------------------------

@contextmanager
def monitored_run(
    *, agent_name: str, kind: str, description: str = "",
    input_text: str = "", external_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Iterator["RunHandle"]:
    """Open a run, hand the caller a handle that lets them append traces,
    and guarantee `finish_run` is called (with status='error' on
    exception, including the traceback in meta_json so debugging is
    possible from the dashboard alone).

    Usage::

        with monitored_run(agent_name="my-agent", kind="ollama",
                           input_text=prompt) as run:
            run.trace("model_call", {"role": "user", "content": prompt})
            reply = ollama_client.chat(...)
            run.trace("model_call", {"role": "assistant",
                                     "content": reply["message"]["content"]})
            run.finish(reply["message"]["content"])
    """
    t0 = time.time()
    with db.session() as conn:
        agent_id = db.upsert_agent(conn, agent_name, description, kind=kind)
        run_id = db.create_run(
            conn, agent_id, external_id=external_id,
            input_text=input_text, meta=meta or {},
        )
    handle = RunHandle(run_id=run_id, agent_id=agent_id, kind=kind, t0=t0)
    try:
        yield handle
    except Exception as e:
        # capture the full traceback so the user can debug from the UI
        with db.session() as conn:
            db.append_trace(conn, run_id, "log", {
                "level": "error",
                "exception": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            db.finish_run(
                conn, run_id, status="error",
                output_text=handle.output_text or "",
                elapsed_ms=int((time.time() - t0) * 1000),
            )
        raise
    else:
        if not handle._finished:
            with db.session() as conn:
                db.finish_run(
                    conn, run_id, status="done",
                    output_text=handle.output_text or "",
                    elapsed_ms=int((time.time() - t0) * 1000),
                )


class RunHandle:
    """Lightweight handle returned by `monitored_run`.

    Adapters call:
      - `handle.trace(kind, payload)` for each event (model_call, tool, log)
      - `handle.record_tokens(model, tokens_in, tokens_out)` to accumulate
        economics (v1.7+). The last `model` wins for `run.model_id`; tokens
        sum. Cost is computed once at `finish()` via `pricing.compute_cost`.
      - `handle.set_output(text)` or `handle.finish(text)` when done.

    All economics are optional. If an adapter never calls record_tokens
    the run still finishes correctly with NULL tokens / cost (and the UI
    renders '\u2014').
    """

    def __init__(self, *, run_id: int, agent_id: int, kind: str, t0: float):
        self.run_id = run_id
        self.agent_id = agent_id
        self.kind = kind
        self.t0 = t0
        self.output_text: str = ""
        self._finished: bool = False
        # v1.7 economics state (None = "never reported", which differs from 0)
        self._tokens_in: Optional[int] = None
        self._tokens_out: Optional[int] = None
        self._model_id: Optional[str] = None

    def trace(self, event_kind: str, payload: Dict[str, Any]) -> int:
        """Append a trace event. Returns the trace event id."""
        with db.session() as conn:
            return db.append_trace(conn, self.run_id, event_kind, payload)

    def record_tokens(
        self, *, model: Optional[str] = None,
        tokens_in: int = 0, tokens_out: int = 0,
    ) -> None:
        """Accumulate token usage for this run (v1.7).

        Call after every model_call where the provider returned a usage
        block. Tokens are summed across calls; `model` is overwritten
        each time -- if a run mixes models, the *last* model id wins.
        That's a deliberate simplification: per-call cost is in the
        trace event itself, the run-level row is the rollup.
        """
        if tokens_in:
            self._tokens_in = (self._tokens_in or 0) + int(tokens_in)
        if tokens_out:
            self._tokens_out = (self._tokens_out or 0) + int(tokens_out)
        if model:
            self._model_id = model

    def set_output(self, text: str) -> None:
        """Record the final output text without finishing the run yet
        (useful when the adapter wants to do post-processing first)."""
        self.output_text = text

    def finish(self, text: str, *, status: str = "done") -> None:
        """Mark the run terminal and write the output. Idempotent: safe
        to call from within `monitored_run` AND let the context manager
        also close out -- the cm sees `_finished=True` and skips."""
        self.output_text = text
        elapsed_ms = int((time.time() - self.t0) * 1000)
        # Lazy-import pricing to avoid a circular import on module load
        from agent_monitor.pricing import compute_cost
        cost = compute_cost(self._model_id, self._tokens_in, self._tokens_out)
        with db.session() as conn:
            db.finish_run(
                conn, self.run_id, status=status,
                output_text=text, elapsed_ms=elapsed_ms,
                tokens_in=self._tokens_in, tokens_out=self._tokens_out,
                model_id=self._model_id, cost_usd=cost,
            )
        self._finished = True
        # v1.8: opportunistic offensive-pattern classification. Best-
        # effort -- a classifier exception must NEVER fail a finishing
        # run, so we swallow everything and continue.
        try:
            from agent_monitor.classifiers.offensive_patterns import (
                classify_run as _op_classify,
            )
            res = _op_classify(self.run_id)
            with db.session() as conn:
                db.persist_classifier_result(
                    conn, self.run_id,
                    classifier="offensive_patterns",
                    score=res["score"], kind=res["kind"],
                    signals=res["signals"],
                )
        except Exception:
            # Intentionally silent -- the dashboard will show the run
            # without a classifier score, which is the honest fallback.
            pass


__all__ = ["AgentAdapter", "RunHandle", "monitored_run"]
