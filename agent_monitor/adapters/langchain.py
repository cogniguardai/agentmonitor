"""
LangChain adapter -- real-time monitoring via a callback handler.

Usage::

    from agent_monitor.adapters.langchain import AgentMonitorCallback
    chain = ChatPromptTemplate.from_template(...) | ChatOpenAI(...) | StrOutputParser()
    result = chain.invoke(
        {"q": "..."},
        config={"callbacks": [AgentMonitorCallback(agent_name='my-chain')]},
    )

The callback opens a `monitored_run` on `on_chain_start` (or
`on_llm_start` if no chain wraps the call), emits `model_call` events
for every LLM round-trip and `tool` events for every tool invocation,
then finishes the run on `on_chain_end` / `on_chain_error`.

LangChain's callback API is stable across v0.1 -> v0.3, but we don't
take a hard dependency on the package: the class is constructed
without importing langchain, and we only import its `BaseCallbackHandler`
inside `__init__` so users who don't have langchain installed can still
import the AgentMonitor package fine.

Interp availability: False. LangChain wraps closed-weight models in
most production use; even when the underlying model is a local Qwen,
LangChain shields the residual stream from us.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from agent_monitor.adapters import RunHandle, monitored_run


def _import_base():
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError as e:
        raise RuntimeError(
            "AgentMonitorCallback requires `langchain-core`. "
            "Install with: pip install langchain-core"
        ) from e
    return BaseCallbackHandler


def make_agent_monitor_callback(
    *, agent_name: str, external_id: Optional[str] = None,
    description: str = "",
) -> Any:
    """Factory: returns an instance of a langchain BaseCallbackHandler
    subclass. We build the subclass at call time so importing this
    module does not require langchain to be installed.
    """
    Base = _import_base()

    class _AgentMonitorCallback(Base):  # type: ignore[misc, valid-type]
        kind: str = "langchain"
        interp_available: bool = False

        def __init__(self) -> None:
            super().__init__()
            self._handle: Optional[RunHandle] = None
            self._cm = None
            self._t0: float = 0.0

        # -------- chain-level hooks --------
        def on_chain_start(self, serialized: Dict[str, Any],
                           inputs: Dict[str, Any], **kwargs: Any) -> None:
            # First chain-level callback opens the run; nested chains
            # are recorded as trace events, not new runs.
            if self._handle is None:
                input_text = _flatten_input(inputs)
                self._t0 = time.time()
                self._cm = monitored_run(
                    agent_name=agent_name, kind=self.kind,
                    description=description or "LangChain chain",
                    input_text=input_text, external_id=external_id,
                    meta={"runtime": self.kind,
                          "chain_name": (serialized or {}).get("name")},
                )
                self._handle = self._cm.__enter__()
            else:
                self._handle.trace("log", {
                    "phase": "nested_chain_start",
                    "name": (serialized or {}).get("name"),
                })

        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
            # Only the outermost chain's end finalises the run.
            if self._handle is not None and not self._handle._finished:
                text = _flatten_output(outputs)
                # Only finish if this is the outer chain (no parent_run_id)
                parent = kwargs.get("parent_run_id")
                if parent is None:
                    self._handle.finish(text)
                    self._cm.__exit__(None, None, None)
                    self._cm = None

        def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
            if self._handle is not None:
                self._handle.trace("log", {
                    "level": "error", "phase": "chain",
                    "message": f"{type(error).__name__}: {error}",
                })
                # Let the context manager record status=error.
                parent = kwargs.get("parent_run_id")
                if parent is None and self._cm is not None:
                    try:
                        self._cm.__exit__(type(error), error, None)
                    except BaseException:
                        pass
                    self._cm = None

        # -------- LLM-level hooks --------
        def on_llm_start(self, serialized: Dict[str, Any],
                         prompts: List[str], **kwargs: Any) -> None:
            # If a chain didn't wrap us, open the run here.
            if self._handle is None:
                self._t0 = time.time()
                self._cm = monitored_run(
                    agent_name=agent_name, kind=self.kind,
                    description=description or "LangChain LLM",
                    input_text=(prompts[0] if prompts else ""),
                    external_id=external_id,
                    meta={"runtime": self.kind,
                          "llm_name": (serialized or {}).get("name")},
                )
                self._handle = self._cm.__enter__()
            self._handle.trace("model_call", {
                "direction": "request",
                "model": (serialized or {}).get("name"),
                "prompts": prompts,
                "invocation_params": kwargs.get("invocation_params"),
            })

        def on_chat_model_start(self, serialized: Dict[str, Any],
                                messages: List[List[Any]], **kwargs: Any) -> None:
            if self._handle is None:
                self._t0 = time.time()
                self._cm = monitored_run(
                    agent_name=agent_name, kind=self.kind,
                    description=description or "LangChain chat",
                    input_text=_last_user_message(messages),
                    external_id=external_id,
                    meta={"runtime": self.kind,
                          "llm_name": (serialized or {}).get("name")},
                )
                self._handle = self._cm.__enter__()
            self._handle.trace("model_call", {
                "direction": "request",
                "model": (serialized or {}).get("name"),
                "messages": _msgs_to_jsonable(messages),
                "invocation_params": kwargs.get("invocation_params"),
            })

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            if self._handle is None:
                return
            generations = getattr(response, "generations", None) or []
            texts: List[str] = []
            for gen_list in generations:
                for gen in gen_list:
                    texts.append(getattr(gen, "text", str(gen)))
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
            self._handle.trace("model_call", {
                "direction": "response",
                "content": "\n".join(texts),
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
            })

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            if self._handle is not None:
                self._handle.trace("log", {
                    "level": "error", "phase": "llm",
                    "message": f"{type(error).__name__}: {error}",
                })

        # -------- tool hooks --------
        def on_tool_start(self, serialized: Dict[str, Any],
                          input_str: str, **kwargs: Any) -> None:
            if self._handle is not None:
                self._handle.trace("tool", {
                    "phase": "started",
                    "name": (serialized or {}).get("name"),
                    "arguments": input_str,
                })

        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            if self._handle is not None:
                self._handle.trace("tool", {
                    "phase": "finished",
                    "result": str(output)[:4000],
                })

        def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
            if self._handle is not None:
                self._handle.trace("tool", {
                    "phase": "error",
                    "message": f"{type(error).__name__}: {error}",
                })

    return _AgentMonitorCallback()


# Backwards-friendly alias matching the rest of the adapter naming.
AgentMonitorCallback = make_agent_monitor_callback


# ---------------------------------------------------------------------
# Internal: tiny helpers to flatten LC inputs/messages into UI-friendly
# strings without dragging in the full langchain.schema namespace.
# ---------------------------------------------------------------------

def _flatten_input(inputs: Any) -> str:
    if isinstance(inputs, str):
        return inputs
    if isinstance(inputs, dict):
        # Common shape: {"input": "..."} or {"q": "..."}
        for k in ("input", "question", "q", "prompt"):
            if k in inputs and isinstance(inputs[k], str):
                return inputs[k]
        try:
            import json
            return json.dumps(inputs, default=str)[:4000]
        except Exception:
            return str(inputs)[:4000]
    return str(inputs)[:4000]


def _flatten_output(outputs: Any) -> str:
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, dict):
        for k in ("output", "text", "answer", "result"):
            if k in outputs and isinstance(outputs[k], str):
                return outputs[k]
        try:
            import json
            return json.dumps(outputs, default=str)[:4000]
        except Exception:
            return str(outputs)[:4000]
    return str(outputs)[:4000]


def _last_user_message(messages: List[List[Any]]) -> str:
    # messages is list-of-list[BaseMessage]; grab the last human message
    for msg_list in reversed(messages or []):
        for m in reversed(msg_list or []):
            content = getattr(m, "content", None)
            if content:
                return str(content)[:4000]
    return ""


def _msgs_to_jsonable(messages: List[List[Any]]) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    for msg_list in messages or []:
        row: List[Dict[str, Any]] = []
        for m in msg_list or []:
            row.append({
                "type": getattr(m, "type", type(m).__name__),
                "content": getattr(m, "content", str(m)),
            })
        out.append(row)
    return out


__all__ = ["AgentMonitorCallback", "make_agent_monitor_callback"]
