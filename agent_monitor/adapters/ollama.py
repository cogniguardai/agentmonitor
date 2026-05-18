"""
Ollama adapter -- monitor any local Ollama model (qwen, llama, mistral,
deepseek, gpt-oss, ...) inside AgentMonitor.

Ollama is the simplest non-Qwen-vLLM runtime to support because it's
already running on the user's box (we use it for NLA decoding and code
scanning), so this adapter has zero new dependencies.

Interp availability
-------------------
Ollama serves many model families. Latent probes are only available for
the Qwen family that the in-house probes were trained on. So even
though Ollama-with-Qwen *technically* has the right architecture, this
adapter conservatively reports `interp_available=False` -- our probes
are wired to vLLM hidden-state extraction, not Ollama's HTTP API. If
you want interp on a local Qwen model, use the existing vLLM agent
runner.

This is the honest call: better to say "no interp here" than to silently
skip it and have the UI look broken.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent_monitor.adapters import monitored_run


DEFAULT_HOST = "http://localhost:11434"


class OllamaAdapter:
    """Wraps Ollama's `POST /api/chat` endpoint into the AgentMonitor
    trace schema.

    Each `.run()` call records:
      - run row (input_text = the user prompt or last user message)
      - one `model_call` trace event for the request
      - one `model_call` trace event per assistant reply
      - any `tool` events if the model emits tool_calls
      - finish_run with the final assistant text + elapsed_ms
    """

    kind: str = "ollama"
    description: str = "Ollama (local)"
    interp_available: bool = False

    def __init__(self, *, model: str, host: str = DEFAULT_HOST,
                 timeout_s: float = 180.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    # -----------------------------------------------------------------
    # The public entrypoint adapters expose
    # -----------------------------------------------------------------
    def run(
        self, input_text: str, *,
        agent_name: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        external_id: Optional[str] = None,
        temperature: float = 0.0,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send `input_text` as the next user message and record a run.

        `history` is an optional list of prior {role, content} dicts.
        `system` is a system prompt prepended once if provided.
        Returns {"run_id": int, "output_text": str, "status": str}.
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
            "host": self.host,
            "temperature": temperature,
        }

        with monitored_run(
            agent_name=agent_name, kind=self.kind,
            description=f"Ollama: {self.model}",
            input_text=input_text, external_id=external_id, meta=meta,
        ) as run:
            # Record the request as a trace event so the Live panel can
            # show what we sent (full conversation, not just the last
            # user message).
            run.trace("model_call", {
                "direction": "request",
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            })

            t0 = time.time()
            try:
                reply = self._chat(messages, temperature, extra_options)
            except Exception as e:
                # Let monitored_run capture the traceback. Re-raise so
                # the caller can react if they want.
                run.trace("log", {
                    "level": "error",
                    "phase": "ollama_http",
                    "message": f"{type(e).__name__}: {e}",
                })
                raise
            latency_ms = int((time.time() - t0) * 1000)

            assistant_msg = reply.get("message") or {}
            assistant_text = (assistant_msg.get("content") or "").strip()

            run.trace("model_call", {
                "direction": "response",
                "model": self.model,
                "role": "assistant",
                "content": assistant_text,
                "latency_ms": latency_ms,
                # surfaced for the UI; full payload also kept for audit
                "tokens_in": reply.get("prompt_eval_count"),
                "tokens_out": reply.get("eval_count"),
            })

            # Tool calls (newer Ollama / Llama-3.1 / Qwen-2.5 models)
            for tc in (assistant_msg.get("tool_calls") or []):
                run.trace("tool", {
                    "phase": "model_proposed",
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                })

            run.finish(assistant_text)
            return {
                "run_id": run.run_id,
                "output_text": assistant_text,
                "status": "done",
                "latency_ms": latency_ms,
            }

    # -----------------------------------------------------------------
    # Internal: HTTP call to Ollama. We use urllib instead of httpx to
    # avoid pulling another dep into the frozen exe.
    # -----------------------------------------------------------------
    def _chat(self, messages: List[Dict[str, str]], temperature: float,
              extra_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        if extra_options:
            body["options"].update(extra_options)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.host}/api/chat",
            data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # Surface the body so users can see "model not pulled" etc.
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"ollama HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"ollama unreachable at {self.host}: {e.reason}. "
                "Is the Ollama daemon running?"
            ) from e
        return json.loads(raw)


__all__ = ["OllamaAdapter"]
