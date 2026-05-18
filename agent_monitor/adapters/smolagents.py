"""
smolagents adapter -- post-hoc trace ingest from `agent.memory.steps`.

WHY POST-HOC
============

smolagents (HuggingFace) exposes a clean `agent.memory.steps` attribute
after a run, containing the full ReAct-style trajectory: action steps,
tool calls, observations, and the final answer. That's a reliable
contract -- the in-process callback API is less so. So we ingest from
memory after `agent.run(...)` returns.

Usage::

    from smolagents import CodeAgent, HfApiModel
    from agent_monitor.adapters.smolagents import record_agent_run

    agent = CodeAgent(tools=[...], model=HfApiModel())
    answer = agent.run("how many seconds in a year?")
    record_agent_run(agent_name="math-agent", agent=agent,
                     input_text="how many seconds in a year?",
                     output_text=str(answer))

Interp availability: False. Same reason as the OpenAI/Anthropic
adapters -- smolagents typically uses HF Inference API or other hosted
models; even when wired to a local Qwen we go through their
chat-template abstraction, not the residual stream.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_monitor.adapters import monitored_run


def record_agent_run(
    *, agent_name: str, agent: Any,
    input_text: str, output_text: str = "",
    external_id: Optional[str] = None,
    description: str = "smolagents CodeAgent",
    elapsed_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk `agent.memory.steps` and ingest the trajectory.

    We duck-type the steps: any object with `model_output`,
    `tool_calls`, `observations`, or `action_output` attributes (or
    dict keys) is recognised. This keeps us version-tolerant.

    Returns {"run_id": int, "status": "done", "n_steps": int}.
    """
    steps = _get_steps(agent)

    meta = {
        "runtime": "smolagents",
        "model_id": _safe_attr(agent, "model"),
        "agent_class": type(agent).__name__,
        "n_steps": len(steps),
        "post_hoc": True,
    }
    with monitored_run(
        agent_name=agent_name, kind="smolagents",
        description=description,
        input_text=input_text, external_id=external_id, meta=meta,
    ) as run:
        for i, step in enumerate(steps):
            # Try to surface the model's thought / output text
            model_output = _safe_get(step, "model_output", "model_output_message")
            if model_output is not None:
                run.trace("model_call", {
                    "direction": "response",
                    "step_index": i,
                    "step_type": type(step).__name__,
                    "content": _stringify(model_output),
                })

            # Tool calls from this step
            for tc in (_safe_get(step, "tool_calls") or []):
                run.trace("tool", {
                    "phase": "called",
                    "step_index": i,
                    "name": _safe_get(tc, "name", "tool_name"),
                    "arguments": _safe_get(tc, "arguments", "tool_arguments"),
                })

            # Observations / tool outputs
            obs = _safe_get(step, "observations", "tool_output", "action_output")
            if obs is not None:
                run.trace("tool", {
                    "phase": "result",
                    "step_index": i,
                    "result": _stringify(obs)[:4000],
                })

            # Errors
            err = _safe_get(step, "error", "error_message")
            if err:
                run.trace("log", {
                    "level": "error",
                    "step_index": i,
                    "message": _stringify(err),
                })

        run.finish(output_text or _last_step_output(steps))

        if elapsed_ms is not None:
            from agent_monitor import db
            with db.session() as conn:
                conn.execute(
                    "UPDATE run SET elapsed_ms = ? WHERE id = ?",
                    (int(elapsed_ms), run.run_id),
                )
        return {
            "run_id": run.run_id,
            "status": "done",
            "n_steps": len(steps),
        }


# ---------------------------------------------------------------------
# Internal helpers -- duck-typing instead of taking a hard smolagents dep
# ---------------------------------------------------------------------

def _get_steps(agent: Any) -> List[Any]:
    mem = getattr(agent, "memory", None)
    if mem is None:
        return []
    steps = getattr(mem, "steps", None)
    if steps is None and isinstance(mem, dict):
        steps = mem.get("steps")
    return list(steps or [])


def _safe_get(obj: Any, *names: str) -> Any:
    """Try attribute access first (smolagents objects), then dict access."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return None


def _safe_attr(obj: Any, name: str) -> Optional[str]:
    v = getattr(obj, name, None)
    if v is None:
        return None
    # If it's a Model instance, surface its repr; if it's a string, return as-is
    return getattr(v, "model_id", None) or str(v)


def _stringify(x: Any) -> str:
    if isinstance(x, str):
        return x
    return str(x)


def _last_step_output(steps: List[Any]) -> str:
    for s in reversed(steps):
        out = _safe_get(s, "action_output", "tool_output", "model_output")
        if out is not None:
            return _stringify(out)
    return ""


__all__ = ["record_agent_run"]
