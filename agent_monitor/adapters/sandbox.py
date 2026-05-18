"""
Generic sandbox-result ingest -- record the outcome of running an
arbitrary workload in an isolated environment.

WHY THIS EXISTS
===============

Many AI / agent workflows include a "run something risky in a sandbox"
step:

  * Run untrusted user code in a Docker / Firecracker / WASM sandbox
    and report exit code, stdout summary, signals.
  * Replay a recorded HTTP trace against a staged service.
  * Smoke-boot a CI artifact in a VM and check it doesn't crash.
  * (Yes, also: load a binary in an instrumented VM and observe what
    happens. We are deliberately domain-agnostic.)

AgentMonitor wants to *observe* these outcomes without having opinions
about them. We persist:

  * What was run (a label + an opaque hashed-input fingerprint)
  * What happened (exit code, duration, an outcome enum)
  * A free-form `signals` dict the caller fills out

We do NOT classify whether the outcome looks malicious, exploitable,
or otherwise interesting. That's the user's job; the classifier in
`agent_monitor.classifiers.offensive_patterns` (item 6) reads these
events and decides, but lives behind a separate explicit feature flag.

WHAT THIS IS NOT
================

* Not a sandbox harness. We do not start containers / VMs / WASM
  instances. The user runs their workload and tells us what happened.
* Not a crash-discriminator. We record outcomes verbatim; we do NOT
  decide whether a crash was "interesting" or "exploitable."
* Not exploit-aware. The `signals` dict is opaque text/numbers from
  the caller's perspective; we never infer primitives.

USAGE
=====

::

    from agent_monitor.adapters.sandbox import record_sandbox_run

    record_sandbox_run(
        agent_name="ci-smoke",
        workload="docker run --rm myapp:pr-1234 ./tests/smoke.sh",
        outcome="pass",         # one of: pass | fail | error | timeout | skipped
        exit_code=0,
        elapsed_ms=14_320,
        signals={
            "stdout_lines":   1024,
            "stderr_lines":   3,
            "memory_peak_mb": 412,
        },
        inputs_hash="sha256:abc...",
    )

For a multi-step sandbox session use the `Sandbox` context manager:

::

    from agent_monitor.adapters.sandbox import Sandbox

    with Sandbox(agent_name="my-eval-harness") as sb:
        sb.run("setup",   outcome="pass", elapsed_ms=120)
        sb.run("test_a",  outcome="pass", exit_code=0, elapsed_ms=400)
        sb.run("test_b",  outcome="fail", exit_code=1, elapsed_ms=2200,
               signals={"failed_assertions": ["x == y"]})
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from agent_monitor.adapters import monitored_run, RunHandle

_VALID_OUTCOMES = ("pass", "fail", "error", "timeout", "skipped", "crash", "unknown")


def _normalize_outcome(s: Optional[str]) -> str:
    if not s:
        return "unknown"
    s = str(s).strip().lower()
    if s in _VALID_OUTCOMES:
        return s
    return {
        "ok":        "pass",
        "success":   "pass",
        "succeeded": "pass",
        "failure":   "fail",
        "failed":    "fail",
        "errored":   "error",
        "exception": "error",
        "timed_out": "timeout",
        "skip":      "skipped",
        "segfault":  "crash",
        "sigsegv":   "crash",
    }.get(s, "unknown")


# ---------------------------------------------------------------------------
# Single-shot helper
# ---------------------------------------------------------------------------

def record_sandbox_run(
    *, agent_name: str, workload: str, outcome: str,
    exit_code: Optional[int] = None,
    elapsed_ms: Optional[int] = None,
    signals: Optional[Dict[str, Any]] = None,
    inputs_hash: Optional[str] = None,
    label: str = "",
    external_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record one sandbox execution as its own short-lived run.

    Args:
        agent_name: AgentMonitor agent name (becomes a row in `agent`).
        workload:   short label for what was executed (a command line
                    summary, a test name, etc.). Stored in `input_text`.
        outcome:    one of pass | fail | error | timeout | skipped |
                    crash | unknown. Variants are normalised.
        exit_code:  optional process exit code.
        elapsed_ms: optional measured duration. If None, the wall-clock
                    duration of the wrapping `monitored_run` is used.
        signals:    free-form dict the caller fills out. Stored verbatim.
        inputs_hash: optional digest of the inputs (we don't store the
                    raw inputs). Use this for sensitive workloads.
        label:      human-readable description for the UI.

    Returns: {"run_id", "outcome", "exit_code", "elapsed_ms", ...}
    """
    o = _normalize_outcome(outcome)
    sig = dict(signals or {})
    with monitored_run(
        agent_name=agent_name, kind="sandbox",
        description=label or "sandbox execution",
        input_text=workload[:4000],
        external_id=external_id,
        meta={"sandbox": True, "inputs_hash": inputs_hash},
    ) as run:
        run.trace("sandbox_result", {
            "workload":    workload[:4000],
            "outcome":     o,
            "exit_code":   exit_code,
            "elapsed_ms":  elapsed_ms,
            "signals":     sig,
            "inputs_hash": inputs_hash,
            "label":       label,
        })
        # If the caller reports an explicit elapsed_ms different from
        # wall-clock, surface it in the run output for quick scanning.
        run.finish(
            f"outcome={o}"
            + (f" exit={exit_code}" if exit_code is not None else "")
            + (f" {elapsed_ms}ms" if elapsed_ms is not None else "")
        )
        return {
            "run_id":     run.run_id,
            "outcome":    o,
            "exit_code":  exit_code,
            "elapsed_ms": elapsed_ms,
        }


# ---------------------------------------------------------------------------
# Context manager (multi-step session)
# ---------------------------------------------------------------------------

@contextmanager
def Sandbox(
    *, agent_name: str, label: str = "sandbox session",
    inputs_hash: Optional[str] = None,
    external_id: Optional[str] = None,
) -> Iterator["SandboxHandle"]:
    """Open ONE run that spans a multi-step sandbox session. Each call
    to `handle.run(...)` adds a `sandbox_result` trace event."""
    with monitored_run(
        agent_name=agent_name, kind="sandbox", description=label,
        input_text=label[:4000], external_id=external_id,
        meta={"sandbox": True, "inputs_hash": inputs_hash, "session": True},
    ) as run:
        handle = SandboxHandle(run, inputs_hash=inputs_hash)
        try:
            yield handle
        finally:
            handle._finalize()


class SandboxHandle:
    def __init__(self, run: RunHandle, *, inputs_hash: Optional[str]):
        self.run = run
        self.inputs_hash = inputs_hash
        self.steps: List[Dict[str, Any]] = []
        self._t_start = time.time()

    def run_step(
        self, name: str, *, outcome: str,
        exit_code: Optional[int] = None,
        elapsed_ms: Optional[int] = None,
        signals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        o = _normalize_outcome(outcome)
        payload = {
            "step_index":  len(self.steps),
            "name":        name,
            "outcome":     o,
            "exit_code":   exit_code,
            "elapsed_ms":  elapsed_ms,
            "signals":     dict(signals or {}),
            "inputs_hash": self.inputs_hash,
        }
        self.run.trace("sandbox_result", payload)
        self.steps.append(payload)
        return payload

    # Friendlier alias matching the docstring
    run = run_step  # type: ignore[assignment]

    def _finalize(self) -> None:
        if self.run._finished:
            return
        n = len(self.steps)
        passed = sum(1 for s in self.steps if s["outcome"] == "pass")
        self.run.finish(f"{passed}/{n} steps passed")


__all__ = ["record_sandbox_run", "Sandbox", "SandboxHandle"]
