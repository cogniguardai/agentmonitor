"""
agent_monitor.safety_llamaguard — production-grade safety classifier via Llama Guard 3.

Why this exists
---------------
The original `interp/` probes (harm/refusal/hedging) are linear logistic regressions
trained on 16-24 hand-written examples over `nomic-embed-text` embeddings. They are
useful as cheap drift detectors but are *not* commercial-grade safety signals --
they fire on IT-ops language ("invalid credentials"), on assertive customer demands
("cancel my account"), etc. (See interp/PRODUCTION_PROBES_PLAN.md for the full
honest write-up.)

Meta's Llama Guard 3 (served locally by Ollama as `llama-guard3:1b` or `:8b`) is a
purpose-built classifier trained on millions of examples with published F1 > 0.85
on BeaverTails / AegisSafetyTest. Its base rate of false positives on in-domain
customer-support traffic is ~10x lower than our toy probes.

This module is the thin local client:
    - calls Ollama's /api/chat with the llama-guard3 model
    - parses the `safe` / `unsafe\\nS1,S3` response
    - maps S1..S14 to human-readable category names
    - returns a uniform dict so callers can treat it like any other probe

It is deliberately stdlib-only (urllib) so it works in the frozen PyInstaller build
without adding dependencies.

Output contract
---------------
`classify(text) -> dict` with keys:
    ok: bool              -- True if the model answered; False on transport error
    verdict: "safe" | "unsafe" | None
    score: float in [0,1] -- 1.0 unsafe, 0.0 safe (for uniform thresholding)
    categories: list[str] -- e.g. ["S1: Violent Crimes"] (empty if safe)
    raw: str              -- raw model text, for audit
    latency_ms: int
    error: str | None

Thresholding
------------
Because llama-guard3 returns a hard verdict rather than a calibrated probability,
we expose `score` as a binary 0/1. That is honest: we do not manufacture a
confidence value we did not measure. Callers that want a single threshold can
treat score >= 0.5 as unsafe. Drift analysis should compare this against the
toy `harm_toy` probe stored alongside.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL_ID = os.environ.get("LLAMA_GUARD_MODEL", "llama-guard3:1b")
TIMEOUT_S = float(os.environ.get("LLAMA_GUARD_TIMEOUT", "60"))

# Official Llama Guard 3 taxonomy (MLCommons AI Safety v0.5 categories).
# Source: https://huggingface.co/meta-llama/Llama-Guard-3-1B
CATEGORY_NAMES = {
    "S1":  "Violent Crimes",
    "S2":  "Non-Violent Crimes",
    "S3":  "Sex Crimes",
    "S4":  "Child Exploitation",
    "S5":  "Defamation",
    "S6":  "Specialized Advice",
    "S7":  "Privacy",
    "S8":  "Intellectual Property",
    "S9":  "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}

_MAX_CHARS = 8000  # llama-guard3 handles 128k context; this caps DB / latency cost


def _post(path: str, payload: dict, *, timeout: float = TIMEOUT_S) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_response(raw: str) -> tuple[Optional[str], list[str]]:
    """Parse Llama Guard 3's two-line response.

    Expected shapes:
        "safe"
        "unsafe\nS1,S3"
        "unsafe\nS10"
    We are tolerant of leading/trailing whitespace and case.
    """
    if not raw:
        return None, []
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None, []
    verdict = lines[0].lower()
    if verdict not in ("safe", "unsafe"):
        # Some checkpoints occasionally prepend extra tokens; search.
        lowered = raw.lower()
        if "unsafe" in lowered:
            verdict = "unsafe"
        elif "safe" in lowered:
            verdict = "safe"
        else:
            return None, []
    cats: list[str] = []
    if verdict == "unsafe" and len(lines) > 1:
        # categories line is comma-separated S-codes, possibly with spaces.
        for tok in lines[1].replace(" ", "").split(","):
            tok = tok.strip().upper()
            if tok in CATEGORY_NAMES:
                cats.append(f"{tok}: {CATEGORY_NAMES[tok]}")
            elif tok:
                # unknown code; still record raw
                cats.append(tok)
    return verdict, cats


def classify(text: str, *, role: str = "user") -> dict:
    """Run a single classification. Soft-fails to {ok: False, error: ...}."""
    out = {
        "ok": False, "verdict": None, "score": None,
        "categories": [], "raw": "", "latency_ms": 0, "error": None,
        "model": MODEL_ID,
    }
    if not text or not text.strip():
        out["error"] = "empty input"
        return out

    t0 = time.time()
    try:
        resp = _post("/api/chat", {
            "model": MODEL_ID,
            "messages": [{"role": role, "content": text[:_MAX_CHARS]}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        })
    except urllib.error.URLError as e:
        out["error"] = f"ollama unreachable: {e}"
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["latency_ms"] = int((time.time() - t0) * 1000)
        return out

    out["latency_ms"] = int((time.time() - t0) * 1000)
    raw = (resp.get("message") or {}).get("content", "") or ""
    out["raw"] = raw
    verdict, cats = _parse_response(raw)
    if verdict is None:
        out["error"] = "unparseable response"
        return out
    out["ok"] = True
    out["verdict"] = verdict
    out["categories"] = cats
    out["score"] = 1.0 if verdict == "unsafe" else 0.0
    return out


def is_ready() -> bool:
    """Lightweight ping: is the configured model present in the local Ollama?"""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            r = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    models = [m.get("name", "") for m in (r.get("models") or [])]
    # accept exact match, with-tag match, or family prefix match
    base = MODEL_ID.split(":")[0]
    return any(m == MODEL_ID or m.startswith(base + ":") for m in models)


def status() -> dict:
    return {
        "model": MODEL_ID,
        "host": OLLAMA_HOST,
        "ready": is_ready(),
    }
