"""
Anthropic Messages adapter.

Monitors any chat call against Anthropic's Messages API
(claude-3.5-sonnet, claude-3.5-haiku, claude-3-opus, ...).

Auth: reads `ANTHROPIC_API_KEY` from the environment, OR the caller
passes `api_key=`. Fails fast if neither is set.

Interp availability: False. Anthropic is closed-weight; we cannot hook
the residual stream.

API shape note: Anthropic is *not* OpenAI-compatible. The role/content
model is the same, but the system prompt is a top-level field (not a
message), and there's no `choices[]` -- the response has a top-level
`content` list of blocks. We handle that here.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent_monitor.adapters import monitored_run


DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicAdapter:
    kind: str = "anthropic"
    description: str = "Anthropic Messages"
    interp_available: bool = False

    def __init__(
        self, *, model: str,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.timeout_s = timeout_s
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "AnthropicAdapter requires an API key. Pass api_key=... "
                "or set the ANTHROPIC_API_KEY environment variable."
            )

    def run(
        self, input_text: str, *,
        agent_name: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        external_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send one user message and record a monitored run."""
        messages: List[Dict[str, Any]] = []
        if history:
            # Anthropic accepts the same role/content shape for history.
            messages.extend(history)
        messages.append({"role": "user", "content": input_text})

        meta = {
            "runtime": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": temperature,
        }
        with monitored_run(
            agent_name=agent_name, kind=self.kind,
            description=f"Anthropic: {self.model}",
            input_text=input_text, external_id=external_id, meta=meta,
        ) as run:
            run.trace("model_call", {
                "direction": "request",
                "model": self.model,
                "system": system,
                "messages": messages,
                "temperature": temperature,
                "tools": tools,
            })

            t0 = time.time()
            try:
                reply = self._chat(
                    messages, system, temperature, max_tokens, tools,
                )
            except Exception as e:
                run.trace("log", {
                    "level": "error", "phase": "anthropic_http",
                    "message": f"{type(e).__name__}: {e}",
                })
                raise
            latency_ms = int((time.time() - t0) * 1000)

            # Anthropic returns a list of content blocks. Concatenate text
            # blocks for the human-readable output; record tool_use blocks
            # as tool trace events.
            blocks = reply.get("content") or []
            text_parts: List[str] = []
            for blk in blocks:
                btype = blk.get("type")
                if btype == "text":
                    text_parts.append(blk.get("text") or "")
                elif btype == "tool_use":
                    run.trace("tool", {
                        "phase": "model_proposed",
                        "id": blk.get("id"),
                        "name": blk.get("name"),
                        "arguments": blk.get("input"),
                    })
            assistant_text = "".join(text_parts)
            usage = reply.get("usage") or {}

            run.trace("model_call", {
                "direction": "response",
                "model": reply.get("model") or self.model,
                "role": "assistant",
                "content": assistant_text,
                "stop_reason": reply.get("stop_reason"),
                "latency_ms": latency_ms,
                "tokens_in": usage.get("input_tokens"),
                "tokens_out": usage.get("output_tokens"),
            })

            run.finish(assistant_text)
            return {
                "run_id": run.run_id,
                "output_text": assistant_text,
                "status": "done",
                "latency_ms": latency_ms,
                "stop_reason": reply.get("stop_reason"),
            }

    def _chat(self, messages, system, temperature, max_tokens, tools) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/messages",
            data=data, method="POST", headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(
                f"anthropic HTTP {e.code}: {detail[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"anthropic unreachable at {self.base_url}: {e.reason}"
            ) from e
        return json.loads(raw)


__all__ = ["AnthropicAdapter"]
