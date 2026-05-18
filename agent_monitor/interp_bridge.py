"""
agent_monitor.interp_bridge — load the production interp probes once, score on demand.

This is the gateway between agent_monitor/ and the existing interp/ module.
Probes are JSON files on disk (~ KB each) but loading them per request would
be wasteful, so we cache them as a process-wide singleton.

Soft-fail: if interp/artifacts/ is missing or any probe fails to load,
score_*() returns None and the caller logs 'NA' instead of crashing the run.
That keeps the monitoring layer robust even when interp is uninitialised.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# In dev, interp/artifacts/ sits next to agent_monitor/ in the workspace.
# When frozen by PyInstaller, the spec copies interp/artifacts/ into
# sys._MEIPASS/interp/artifacts.
if getattr(sys, "frozen", False):
    BASE = Path(getattr(sys, "_MEIPASS", "."))
else:
    BASE = Path(__file__).resolve().parent.parent
ART = BASE / "interp" / "artifacts"


@dataclass
class _Bundle:
    harm: object = None
    refusal: object = None
    hedging: object = None
    embed_one: callable = None
    loaded: bool = False
    error: Optional[str] = None


_BUNDLE: _Bundle = _Bundle()


def _load_once() -> _Bundle:
    global _BUNDLE
    if _BUNDLE.loaded or _BUNDLE.error:
        return _BUNDLE
    try:
        from interp.probes import LinearProbe
        from interp.embeddings import embed_one
        b = _Bundle(embed_one=embed_one)
        for name in ("harm", "refusal", "hedging"):
            p = ART / f"probe_{name}.json"
            if p.exists():
                setattr(b, name, LinearProbe.load(p))
        b.loaded = True
        _BUNDLE = b
    except Exception as e:
        _BUNDLE = _Bundle(error=str(e))
    return _BUNDLE


def is_ready() -> bool:
    b = _load_once()
    return b.loaded and any([b.harm, b.refusal, b.hedging])


def status() -> dict:
    b = _load_once()
    return {
        "loaded": b.loaded,
        "error": b.error,
        "probes": {
            "harm": b.harm is not None,
            "refusal": b.refusal is not None,
            "hedging": b.hedging is not None,
        },
    }


def _safe_score(probe, text: str) -> Optional[float]:
    if probe is None or not text or not text.strip():
        return None
    text = text.strip()[:6000]   # nomic-embed-text 8k token cap
    try:
        b = _load_once()
        vec = b.embed_one(text)
        return float(probe.predict_proba(vec))
    except Exception:
        return None


def score_harm(text: str) -> Optional[float]:
    return _safe_score(_load_once().harm, text)


def score_refusal(text: str) -> Optional[float]:
    return _safe_score(_load_once().refusal, text)


def score_hedging(text: str) -> Optional[float]:
    return _safe_score(_load_once().hedging, text)


def score_all(text: str) -> dict:
    """Return harm/refusal/hedging scores for `text`.

    `harm` is the primary commercial-grade signal from Llama Guard 3 when the
    model is reachable; otherwise it falls back to the toy embedding probe.
    The toy harm probe is *also* always reported as `harm_toy` so it can be
    used as a drift detector regardless of which signal is primary.

    refusal / hedging remain toy probes for now -- they are weak signals used
    to flag content for human review, never as gates.

    Returned dict keys map 1:1 to rows in the `interp_score` table.
    """
    # Always compute the toy probes -- they are cheap (one embed call total
    # because all three share the same vector internally? no: embed cost is
    # paid once per probe call today -- still <100ms locally).
    toy_harm = score_harm(text)
    toy_refusal = score_refusal(text)
    toy_hedging = score_hedging(text)

    # Primary harm signal: Llama Guard 3 (soft-fail to toy if unavailable).
    primary_harm: Optional[float] = None
    primary_source = "toy"
    primary_categories: list = []
    primary_latency_ms: Optional[int] = None
    try:
        from agent_monitor import safety_llamaguard
        result = safety_llamaguard.classify(text)
        if result.get("ok"):
            primary_harm = float(result["score"])
            primary_source = "llama_guard_3"
            primary_categories = result.get("categories") or []
            primary_latency_ms = result.get("latency_ms")
    except Exception:
        # never break a run on safety-classifier failure
        pass

    if primary_harm is None:
        primary_harm = toy_harm  # fallback

    return {
        # primary, used by dashboard "harm" column
        "harm": primary_harm,
        # secondary signals kept for audit + drift detection
        "harm_toy": toy_harm,
        "refusal": toy_refusal,
        "hedging": toy_hedging,
        # metadata (not numeric; runner skips None numeric scores)
        "_meta": {
            "primary_source": primary_source,
            "primary_categories": primary_categories,
            "primary_latency_ms": primary_latency_ms,
        },
    }
