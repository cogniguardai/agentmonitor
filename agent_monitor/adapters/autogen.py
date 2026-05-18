"""
AutoGen adapter -- post-hoc trace ingest.

WHY POST-HOC AND NOT REAL-TIME
==============================

AutoGen has had three substantially different agent APIs in <18 months:
v0.2 ConversableAgent, v0.4 autogen-agentchat, and the autogen-core
event-bus refactor. Hooking into any of those in-process means taking a
hard dep on one specific version and breaking when the user upgrades.

The honest alternative: AutoGen always exposes the full conversation as
a list of `{role, content, name?}` dicts at the end of a run. Call
`record_conversation(...)` once the agents are done. You get every
message in the trace_event table with `kind='model_call'`, plus the run
row with input = first user message and output = last assistant
message. This works on every AutoGen version.

If the user prefers real-time, AutoGen v0.2's `ConversableAgent.register_reply`
hook is straightforward to wire up themselves -- they just call
`handle.trace('model_call', ...)` inside the callback. We document the
pattern but do not ship the wiring.

Interp availability: False. AutoGen runs against external chat APIs,
not residual streams.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agent_monitor.adapters import monitored_run


def record_conversation(
    *, agent_name: str,
    messages: List[Dict[str, Any]],
    model_hint: Optional[str] = None,
    external_id: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    description: str = "AutoGen conversation",
) -> Dict[str, Any]:
    """Ingest a finished AutoGen conversation into AgentMonitor.

    Args:
        agent_name: Logical name shown in the dashboard (e.g.
            "support-team" or "researcher+critic").
        messages: The full conversation as a list of dicts with at
            least `role` and `content` keys. Optional keys recognised:
            `name` (agent that spoke), `tool_calls`, `tool_call_id`.
            Pass `groupchat.messages` for a GroupChat run, or the
            return value of `initiate_chat` for a 2-agent run.
        model_hint: Optional model id (e.g. "gpt-4o") for the meta row.
            We can't infer it from the messages alone.
        external_id: User-supplied stable id for the conversation
            (e.g. ticket id), surfaced in the runs list.
        elapsed_ms: If you measured wall time yourself, pass it; we use
            it instead of "(time we recorded) - (time we started)" which
            would be ~0 for post-hoc ingest.

    Returns: {"run_id": int, "status": "done", "n_messages": int}.
    """
    msgs = list(messages or [])

    # input_text = first user (or earliest non-system) message
    input_text = ""
    for m in msgs:
        role = (m.get("role") or "").lower()
        if role == "user":
            input_text = str(m.get("content") or "")
            break
    if not input_text:
        for m in msgs:
            if (m.get("role") or "").lower() != "system":
                input_text = str(m.get("content") or "")
                break

    # output_text = last assistant message
    output_text = ""
    for m in reversed(msgs):
        if (m.get("role") or "").lower() == "assistant":
            output_text = str(m.get("content") or "")
            break
    if not output_text and msgs:
        output_text = str(msgs[-1].get("content") or "")

    meta = {
        "runtime": "autogen",
        "model_hint": model_hint,
        "n_messages": len(msgs),
        "post_hoc": True,
    }
    with monitored_run(
        agent_name=agent_name, kind="autogen",
        description=description,
        input_text=input_text, external_id=external_id, meta=meta,
    ) as run:
        for i, m in enumerate(msgs):
            payload: Dict[str, Any] = {
                "direction": "response" if (m.get("role") or "").lower() == "assistant" else "request",
                "role": m.get("role"),
                "name": m.get("name"),
                "content": m.get("content"),
            }
            if m.get("tool_calls"):
                payload["tool_calls"] = m["tool_calls"]
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    run.trace("tool", {
                        "phase": "model_proposed",
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                        "message_index": i,
                    })
            if m.get("tool_call_id"):
                payload["tool_call_id"] = m["tool_call_id"]
                run.trace("tool", {
                    "phase": "result",
                    "tool_call_id": m["tool_call_id"],
                    "result": m.get("content"),
                    "message_index": i,
                })
            run.trace("model_call", payload)

        # If the caller measured real elapsed time, persist it; else
        # post-hoc ingest will show near-zero elapsed which would be a lie.
        run.finish(output_text)
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
            "n_messages": len(msgs),
        }


__all__ = ["record_conversation"]
