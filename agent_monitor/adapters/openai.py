"""
OpenAI Chat Completions adapter.

Monitors any chat call against OpenAI's API (gpt-4o, gpt-4o-mini,
gpt-4.1, o1, o3-mini, ...) or any OpenAI-compatible endpoint
(Together, Groq, vLLM-served models, LM Studio, OpenRouter, ...).

Auth: reads `OPENAI_API_KEY` from the environment, OR the caller passes
`api_key=` to the constructor. If no key is available we fail fast at
adapter construction with a clear message instead of leaking a
half-constructed object that crashes mid-run.

Interp availability: False. We have no probes for closed-weights models,
and openai-compatible self-hosted Qwens still go through the HTTP API,
not the residual stream. Honest answer: no.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent_monitor.adapters import monitored_run


DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIAdapter:
    kind: str = "openai"
    description: str = "OpenAI Chat Completions"
    interp_available: bool = False

    def __init__(
        self, *, model: str,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        organization: Optional[str] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.organization = organization
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAIAdapter requires an API key. Pass api_key=... "
                "or set the OPENAI_API_KEY environment variable. "
                "(For OpenAI-compatible self-hosted endpoints any non-"
                "empty string works.)"
            )

    def run(
        self, input_text: str, *,
        agent_name: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        external_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send one user message and record a monitored run.

        `tools` follows the OpenAI tools schema. Any tool_calls in the
        response are recorded as `tool` trace events with phase
        ='model_proposed' (we do not auto-execute -- that's the caller's
        job).
        """
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
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
            description=f"OpenAI: {self.model}",
            input_text=input_text, external_id=external_id, meta=meta,
        ) as run:
            run.trace("model_call", {
                "direction": "request",
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "tools": tools,
            })

            t0 = time.time()
            try:
                reply = self._chat(messages, temperature, max_tokens, tools)
            except Exception as e:
                run.trace("log", {
                    "level": "error", "phase": "openai_http",
                    "message": f"{type(e).__name__}: {e}",
                })
                raise
            latency_ms = int((time.time() - t0) * 1000)

            choice = (reply.get("choices") or [{}])[0]
            assistant_msg = choice.get("message") or {}
            assistant_text = (assistant_msg.get("content") or "") or ""
            usage = reply.get("usage") or {}

            run.trace("model_call", {
                "direction": "response",
                "model": reply.get("model") or self.model,
                "role": "assistant",
                "content": assistant_text,
                "finish_reason": choice.get("finish_reason"),
                "latency_ms": latency_ms,
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
            })

            for tc in (assistant_msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                run.trace("tool", {
                    "phase": "model_proposed",
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                })

            run.finish(assistant_text)
            return {
                "run_id": run.run_id,
                "output_text": assistant_text,
                "status": "done",
                "latency_ms": latency_ms,
                "finish_reason": choice.get("finish_reason"),
            }

    def _chat(self, messages, temperature, max_tokens, tools) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if tools:
            body["tools"] = tools
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
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
                f"openai HTTP {e.code}: {detail[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"openai unreachable at {self.base_url}: {e.reason}"
            ) from e
        return json.loads(raw)


__all__ = ["OpenAIAdapter"]
