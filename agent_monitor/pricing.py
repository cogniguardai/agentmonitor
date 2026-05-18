"""
agent_monitor.pricing -- public list-price lookup for major LLM APIs.

WHAT THIS IS
============

A small, honest, OFFLINE price table. We compute `cost_usd = (tokens_in
* prompt_price + tokens_out * completion_price) / 1_000_000` and persist
it on `run.cost_usd`. When the model isn't in the table we return None
and the UI shows '—' -- we do not guess.

WHY OFFLINE
===========

Live price scraping would add a network dependency and a moving target
that breaks audit trails. List prices change rarely (weeks to months).
The `updated_at` per entry tells the user how stale a given row is, and
the UI exposes a small note. If a price moves and we haven't updated,
the user sees the same cost they'd compute by hand from the model
provider's pricing page -- still useful, never fraudulent.

LOCAL MODELS
============

Local models (Qwen via vLLM, Ollama-hosted models) cost zero USD per
token by design -- the user already paid for the GPU / electricity. We
store cost_usd = 0.0 (NOT NULL) for those, and surface a column note in
the UI explaining the zero is *modelled*, not measured. This is the
right behaviour: a user comparing "ran on local Qwen" vs. "ran on
gpt-4o" should see local as free under this accounting.

ATTRIBUTION
===========

All prices below are taken from each provider's public pricing page at
the date listed. We do not commit to keeping these current -- the user
is welcome to override via the data dir (see `load_price_overrides`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# (prompt_$_per_1M, completion_$_per_1M, updated_at, source_url)
# Keep keys lower-case; we match case-insensitively.
_DEFAULT_PRICES: Dict[str, Tuple[float, float, str, str]] = {
    # --- OpenAI (https://openai.com/api/pricing) -----------------------------
    "gpt-4o":               (2.50,  10.00, "2025-09-01", "https://openai.com/api/pricing"),
    "gpt-4o-mini":          (0.15,  0.60,  "2025-09-01", "https://openai.com/api/pricing"),
    "gpt-4o-2024-08-06":    (2.50,  10.00, "2025-09-01", "https://openai.com/api/pricing"),
    "gpt-4-turbo":          (10.00, 30.00, "2025-09-01", "https://openai.com/api/pricing"),
    "gpt-3.5-turbo":        (0.50,  1.50,  "2025-09-01", "https://openai.com/api/pricing"),
    "o1-preview":           (15.00, 60.00, "2025-09-01", "https://openai.com/api/pricing"),
    "o1-mini":              (3.00,  12.00, "2025-09-01", "https://openai.com/api/pricing"),

    # --- Anthropic (https://www.anthropic.com/pricing#anthropic-api) ---------
    "claude-3-5-sonnet-20241022": (3.00,  15.00, "2025-09-01", "https://www.anthropic.com/pricing"),
    "claude-3-5-sonnet-latest":   (3.00,  15.00, "2025-09-01", "https://www.anthropic.com/pricing"),
    "claude-3-5-haiku-latest":    (0.80,  4.00,  "2025-09-01", "https://www.anthropic.com/pricing"),
    "claude-3-opus-20240229":     (15.00, 75.00, "2025-09-01", "https://www.anthropic.com/pricing"),
    "claude-3-sonnet-20240229":   (3.00,  15.00, "2025-09-01", "https://www.anthropic.com/pricing"),
    "claude-3-haiku-20240307":    (0.25,  1.25,  "2025-09-01", "https://www.anthropic.com/pricing"),

    # --- Local models (zero by accounting policy; see docstring) ------------
    # Anything matching these prefixes returns (0.0, 0.0) via _match_local().
    # Listed explicitly here so they show up in /api/pricing.
    "qwen2.5-coder:3b":   (0.0, 0.0, "local", "local-runtime"),
    "qwen2.5-coder:7b":   (0.0, 0.0, "local", "local-runtime"),
    "qwen2.5:7b":         (0.0, 0.0, "local", "local-runtime"),
    "llama-guard3:8b":    (0.0, 0.0, "local", "local-runtime"),
}

# Anything starting with one of these prefixes is treated as a local
# (zero-cost) model. Lets us catch user-tagged variants like
# 'qwen2.5-coder:14b-q4_K_M' without hard-coding every quant.
_LOCAL_PREFIXES = ("qwen", "llama-guard", "llama3", "mistral", "ollama:", "vllm:")


def _match_local(model_id: str) -> bool:
    m = model_id.lower()
    return any(m.startswith(p) for p in _LOCAL_PREFIXES)


def load_price_overrides() -> Dict[str, Tuple[float, float, str, str]]:
    """Read `<data_dir>/pricing_overrides.json` if present.

    Format: `{"model-id": {"prompt": 1.23, "completion": 4.56,
                           "updated_at": "2025-12-01",
                           "source": "https://..."}}`

    The data dir resolution matches `agent_monitor.__init__` (dev:
    package data/, frozen: %LOCALAPPDATA%/AgentMonitor/).
    """
    try:
        from agent_monitor import DATA_DIR  # type: ignore
        path = Path(DATA_DIR) / "pricing_overrides.json"
    except Exception:
        path = Path(os.environ.get("LOCALAPPDATA", ".")) / "AgentMonitor" / "pricing_overrides.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, Tuple[float, float, str, str]] = {}
    for k, v in raw.items():
        try:
            out[k.lower()] = (
                float(v["prompt"]), float(v["completion"]),
                str(v.get("updated_at", "user-override")),
                str(v.get("source", "user-override")),
            )
        except Exception:
            continue
    return out


def get_pricing(model_id: str) -> Optional[Tuple[float, float, str, str]]:
    """Return (prompt_$/1M, completion_$/1M, updated_at, source) or None."""
    if not model_id:
        return None
    key = model_id.lower().strip()
    # Overrides win over defaults
    overrides = load_price_overrides()
    if key in overrides:
        return overrides[key]
    if key in _DEFAULT_PRICES:
        return _DEFAULT_PRICES[key]
    if _match_local(key):
        return (0.0, 0.0, "local", "local-runtime")
    return None


def compute_cost(
    model_id: Optional[str], tokens_in: Optional[int], tokens_out: Optional[int],
) -> Optional[float]:
    """Compute USD cost. Returns None when ANY of (model, tokens_in,
    tokens_out) is missing or model is unknown.

    Intentionally strict: we never round zero up, never extrapolate. A
    None return is the UI's cue to render '—'.
    """
    if not model_id or tokens_in is None or tokens_out is None:
        return None
    p = get_pricing(model_id)
    if p is None:
        return None
    prompt_per_1m, completion_per_1m, _, _ = p
    cost = (int(tokens_in) * prompt_per_1m + int(tokens_out) * completion_per_1m) / 1_000_000.0
    return round(cost, 6)


def list_prices() -> Dict[str, Dict[str, object]]:
    """For /api/pricing -- a flat view of every known model."""
    merged: Dict[str, Tuple[float, float, str, str]] = dict(_DEFAULT_PRICES)
    merged.update(load_price_overrides())
    return {
        k: {
            "prompt_per_1m":     v[0],
            "completion_per_1m": v[1],
            "updated_at":        v[2],
            "source":            v[3],
        }
        for k, v in merged.items()
    }


__all__ = ["compute_cost", "get_pricing", "list_prices", "load_price_overrides"]
