"""
Generic pipeline adapter -- ingest any bespoke multi-step LLM workflow.

WHAT THIS IS
============

Sometimes a user is running their own Python / Bash pipeline that
doesn't fit any of the framework adapters (LangChain, AutoGen, etc.).
They want AgentMonitor's observability surface (trace events, runs,
cost / tokens, the UI) without rewriting the pipeline as a chain.

This module provides a single class, `Pipeline`, that wraps an
arbitrary sequence of steps. Each step is opaque to AgentMonitor: a
name, an optional hashed-input fingerprint, an optional summary of the
output, and token / model accounting. We do not look at the step's
*contents*; we record what the user tells us.

DUAL-USE POSTURE
================

Pipelines can carry sensitive material -- source code under audit,
penetration-test artifacts, internal customer data, research IP. When
the user passes `sensitive=True` we:

  1. Stamp the run with `meta.sensitive = True`.
  2. Never persist `inputs_hash` raw contents (the user is expected
     to hash on their side; AgentMonitor stores the hex digest).
  3. Cap `outputs_summary` to 4000 chars and mark it as a summary
     (the convention is: a short human-readable description, NOT the
     raw output).
  4. The UI shows a prominent amber banner on the run-detail page so
     anyone with dashboard access knows the source was flagged.

This isn't a security boundary -- a determined user can always log
whatever they want into a trace event. It's a *consent surface*: the
flag forces the user to think about retention.

USAGE
=====

::

    from agent_monitor.adapters.pipeline import Pipeline

    with Pipeline(agent_name="my-research-pipeline",
                  sensitive=True) as pipe:

        pipe.step("scrape", outputs_summary="found 412 candidates",
                  model=None, tokens_in=0, tokens_out=0)

        pipe.step("filter",
                  inputs_hash=sha256_of_candidates_list,
                  outputs_summary="kept 87 after signature check")

        pipe.step("llm_score",
                  inputs_hash=sha256_of_filtered_list,
                  outputs_summary="ranked; top score=0.87",
                  model="gpt-4o-mini",
                  tokens_in=12000, tokens_out=900)

        pipe.finish("pipeline complete: 1 winner selected")
"""
from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from agent_monitor.adapters import monitored_run, RunHandle


def hash_content(data: Any) -> str:
    """Convenience: stable SHA-256 hex digest for any utf-8-encodable
    value. Returns the user's pre-computed hash if they pass a 64-char
    hex string."""
    if isinstance(data, str) and len(data) == 64 and all(
        c in "0123456789abcdef" for c in data.lower()
    ):
        return data.lower()
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    s = data if isinstance(data, str) else repr(data)
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


@contextmanager
def Pipeline(
    *, agent_name: str, description: str = "external pipeline",
    sensitive: bool = False, external_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Iterator["PipelineHandle"]:
    """Open a run, hand back a `PipelineHandle` for step recording."""
    meta = dict(meta or {})
    meta["pipeline"] = True
    if sensitive:
        meta["sensitive"] = True
    with monitored_run(
        agent_name=agent_name, kind="pipeline",
        description=description,
        input_text=f"pipeline ({'sensitive' if sensitive else 'open'})",
        external_id=external_id, meta=meta,
    ) as run:
        handle = PipelineHandle(run, sensitive=sensitive)
        try:
            yield handle
        finally:
            handle._finalize()


class PipelineHandle:
    """Lightweight wrapper that records pipeline steps as trace events."""

    def __init__(self, run: RunHandle, *, sensitive: bool):
        self.run = run
        self.sensitive = sensitive
        self.steps: List[Dict[str, Any]] = []
        self._t_start = time.time()

    def step(
        self, name: str, *,
        inputs_hash: Optional[str] = None,
        outputs_summary: str = "",
        model: Optional[str] = None,
        tokens_in: int = 0, tokens_out: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record one pipeline step.

        Args:
            name: short label ("scrape", "filter", "llm_score", ...).
            inputs_hash: hex digest of the step inputs. We do NOT store
                raw inputs -- the user is responsible for hashing.
                Pass None when not applicable.
            outputs_summary: human-readable summary, capped at 4000 chars.
                Avoid raw model outputs when `sensitive=True` -- this is
                a summary surface, not a log surface.
            model: model id, for cost accounting (None for non-LLM steps).
            tokens_in, tokens_out: usage. Aggregated into the run total
                via `run.record_tokens()`.
            extra: arbitrary structured data appended verbatim to the
                trace event. The user owns its contents and retention.

        Returns the recorded step dict.
        """
        if outputs_summary and len(outputs_summary) > 4000:
            outputs_summary = outputs_summary[:4000] + "\u2026[truncated]"
        step_idx = len(self.steps)
        payload: Dict[str, Any] = {
            "step_index":      step_idx,
            "name":            name,
            "inputs_hash":     inputs_hash,
            "outputs_summary": outputs_summary,
            "model":           model,
            "tokens_in":       int(tokens_in or 0),
            "tokens_out":      int(tokens_out or 0),
            "elapsed_ms":      int((time.time() - self._t_start) * 1000),
            "sensitive":       self.sensitive,
        }
        if extra:
            payload["extra"] = extra
        self.run.trace("pipeline_step", payload)
        if model and (tokens_in or tokens_out):
            self.run.record_tokens(model=model,
                                   tokens_in=tokens_in, tokens_out=tokens_out)
        self.steps.append(payload)
        return payload

    def log(self, message: str, *, level: str = "info") -> None:
        """Convenience: append a free-form log event. Same `sensitive`
        rules apply -- don't log raw inputs."""
        self.run.trace("log", {
            "level": level, "message": message[:4000],
            "sensitive": self.sensitive,
        })

    def finish(self, summary: str = "") -> None:
        if self.run._finished:
            return
        n = len(self.steps)
        tail = summary or f"{n} steps completed"
        self.run.finish(tail)

    def _finalize(self) -> None:
        if not self.run._finished:
            self.finish()


__all__ = ["Pipeline", "PipelineHandle", "hash_content"]
