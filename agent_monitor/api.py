"""
agent_monitor.api — FastAPI app: REST endpoints + live trace WebSocket.

Endpoints (all under /api/...):
    GET    /api/status                     overall: ollama, probes, browser
    GET    /api/agents                     list agent classes
    GET    /api/runs?agent_id=&limit=&kind=  recent runs (v1.6: kind filter)
    GET    /api/runs/{run_id}              run details + trace + interp scores
    GET    /api/runs/cost_summary?group_by=kind|model|agent (v1.7)
    GET    /api/pricing                    public LLM list prices (v1.7)
    GET    /api/memory?q=&semantic=&limit= search memory
    POST   /api/memory                     add a memory chunk
    GET    /api/interp/status              probe load status
    POST   /api/interp/score               score arbitrary text
    GET    /api/browser/status             browser session info
    POST   /api/browser/goto               navigate to URL
    POST   /api/browser/close              close session
    GET    /api/browser/screenshot.png     current screenshot (PNG)
    WS     /ws/runs/{run_id}               live trace stream (polls SQLite, emits new events)

Static UI is served from agent_monitor/web/ under "/".
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_monitor import db, __version__ as _VERSION

# ---------------------------------------------------------------------------
# Optional Phase-2 modules (v0.1.0 lazy-loading).
#
# The slim `pip install cogniguardai` baseline does NOT bundle the heavier
# features (interp probes, long-term memory, NLA pipeline, browser
# automation, code scanning). Those modules either depend on PyTorch /
# transformers / Playwright, or rely on Mythos-internal infrastructure
# that isn't redistributable yet.
#
# We import them best-effort here. Any module that fails to import
# becomes None, and every endpoint that touches it goes through
# `_require()` which raises HTTPException(503) with a clear install hint.
# The dashboard JS already renders 503 / `_probe_unavailable` cleanly,
# so the user sees a tidy "feature not in this build" indicator instead
# of an exception trace.
# ---------------------------------------------------------------------------

def _try_import(name: str):
    """Best-effort import. Returns None if the module / its deps are missing."""
    try:
        import importlib
        return importlib.import_module(name)
    except Exception:
        return None


interp_bridge = _try_import("agent_monitor.interp_bridge")
memory        = _try_import("agent_monitor.memory")
nla_client    = _try_import("agent_monitor.nla_client")
nla_cache     = _try_import("agent_monitor.nla_cache")
nla_worker    = _try_import("agent_monitor.nla_worker")
browser_mod   = _try_import("agent_monitor.browser")
code_scan_mod = _try_import("agent_monitor.code_scan")


def _require(mod, feature: str, extra: str = "") -> Any:
    """Return `mod` if available, otherwise raise 503 with an install hint."""
    if mod is None:
        msg = f"The '{feature}' feature is not available in this build."
        if extra:
            msg += f" Install with: pip install 'cogniguardai[{extra}]'"
        else:
            msg += " Coming in a future release."
        raise HTTPException(status_code=503, detail=msg)
    return mod

WEB_DIR = Path(__file__).parent / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="AgentMonitor", version=_VERSION)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _on_startup() -> None:
    db.init_db()
    _start_probe_workers()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    # Stop the background status probes first; they are daemon threads
    # so they'd die with the process anyway, but signalling them lets
    # any in-flight urllib socket close cleanly.
    try:
        _stop_probe_workers(timeout=1.0)
    except Exception:
        pass
    # Drain the NLA worker thread cleanly so the frozen exe exits without
    # leaving zombie threads holding the SQLite connection.
    if nla_worker is not None:
        try:
            nla_worker.shutdown(timeout=2.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MemoryIn(BaseModel):
    text: str
    source: str = "manual"
    kind: str = "note"
    tags: list[str] = []
    with_vector: bool = True


class ScoreIn(BaseModel):
    text: str


class NLADecodeIn(BaseModel):
    text: str
    target: str = "adhoc"            # 'input' | 'output' | 'cot' | 'adhoc'
    run_id: Optional[int] = None     # if provided, decoding is persisted to that run
    target_model: Optional[str] = None  # remote backend hint
    layer: Optional[int] = None         # remote backend hint
    persist: bool = True


class GotoIn(BaseModel):
    url: str
    headless: bool = True


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _ollama_status(host: str = "http://localhost:11434") -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        models = [m.get("name") for m in data.get("models", [])]
        return {"up": True, "models": models}
    except Exception as e:
        return {"up": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
#
# Design (v1.9.1):
#
#   /api/status used to call ollama / interp / nla / browser probes
#   *sequentially*, each with its own multi-second urllib timeout. With
#   all three external backends offline (typical first-run experience),
#   /api/status took ~14s -- making the dashboard feel broken.
#
#   The fix is *not* to call probes in parallel per-request (that just
#   trades sequential blocking for thread-pool saturation under
#   sustained polling). Instead, probes run on dedicated long-lived
#   background threads that refresh a shared cache every
#   _PROBE_INTERVAL_S seconds. /api/status is then a pure cache read:
#   O(1), sub-millisecond, never blocks.
#
#   Cold-start UX: before the first probe completes, the cache holds
#   {"_probe_pending": True} stubs. The dashboard JS already renders
#   unknown shapes as a neutral "checking..." indicator. Within a few
#   seconds of app launch, all 4 cache slots are populated with real
#   data.

# How often each background probe refreshes its cache slot.
_PROBE_INTERVAL_S = 5.0

# Shared cache: probe_name -> probe-result dict. We seed every slot
# with _probe_pending=True so /api/status always returns 4 keys, even
# before the workers have produced their first reading.
_PROBE_CACHE: Dict[str, Dict[str, Any]] = {
    "ollama":  {"_probe_pending": True},
    "interp":  {"_probe_pending": True},
    "nla":     {"_probe_pending": True},
    "browser": {"_probe_pending": True},
}
_PROBE_LOCK = threading.Lock()
_PROBE_STOP = threading.Event()
_PROBE_THREADS: List[threading.Thread] = []


def _probe_browser() -> Dict[str, Any]:
    if browser_mod is None:
        return {"open": False, "last_url": None, "last_title": None,
                "_probe_unavailable": True}
    s = browser_mod._SESSION
    if s is None or not s.is_open:
        return {"open": False, "last_url": None, "last_title": None}
    return {
        "open": True,
        "last_url": getattr(s, "last_url", None),
        "last_title": getattr(s, "last_title", None),
    }


def _probe_worker(name: str, fn) -> None:
    """Long-lived worker: every _PROBE_INTERVAL_S seconds, run `fn()`
    and atomically replace _PROBE_CACHE[name] with its result.

    If the underlying probe wedges (e.g., Ollama at 11434 is unreachable
    and urllib's connect takes 2s to time out), this worker simply
    takes that long to complete its current iteration -- there is no
    queueing problem because each name has its own dedicated thread.
    """
    while not _PROBE_STOP.is_set():
        t0 = time.monotonic()
        try:
            value: Dict[str, Any] = fn() or {}
        except Exception as e:
            value = {"_probe_error": f"{type(e).__name__}: {e}"}
        value["_probe_ts"] = time.time()
        value["_probe_age_ms"] = 0
        value["_probe_elapsed_ms"] = int((time.monotonic() - t0) * 1000)
        with _PROBE_LOCK:
            _PROBE_CACHE[name] = value
        if _PROBE_STOP.wait(timeout=_PROBE_INTERVAL_S):
            return


def _start_probe_workers() -> None:
    """Spawn one background thread per probe. Idempotent.

    Probes for optional Phase-2 modules (interp / nla / browser) are
    skipped when the module is missing. The corresponding /api/status
    cache slot is then populated with a stable {"_probe_unavailable":
    True, ...} stub so the dashboard renders cleanly.
    """
    if _PROBE_THREADS:           # already started
        return
    probes: List[tuple] = [("ollama", _ollama_status)]
    if interp_bridge is not None:
        probes.append(("interp", interp_bridge.status))
    else:
        with _PROBE_LOCK:
            _PROBE_CACHE["interp"] = {"_probe_unavailable": True,
                                      "_probe_ts": time.time()}
    if nla_client is not None:
        probes.append(("nla", nla_client.status))
    else:
        with _PROBE_LOCK:
            _PROBE_CACHE["nla"] = {"_probe_unavailable": True,
                                   "_probe_ts": time.time()}
    # browser probe runs through _probe_browser which already handles
    # the missing-module case, so we always start it.
    probes.append(("browser", _probe_browser))
    for name, fn in probes:
        t = threading.Thread(
            target=_probe_worker, args=(name, fn),
            name=f"status-probe-{name}", daemon=True,
        )
        t.start()
        _PROBE_THREADS.append(t)


def _stop_probe_workers(timeout: float = 1.0) -> None:
    """Signal all probe threads to exit at their next interval boundary.
    Threads are daemons so we don't *have* to join, but we make a best
    effort so test runs don't leak."""
    _PROBE_STOP.set()
    for t in _PROBE_THREADS:
        t.join(timeout=timeout)
    _PROBE_THREADS.clear()


@app.get("/api/status")
def api_status() -> Dict[str, Any]:
    """Aggregate health snapshot. O(1) cache read -- see module header.

    Returns a dict with exactly four keys (ollama, interp, nla, browser).
    Each value is either a real probe result, a {"_probe_pending": true}
    stub (before the first refresh completes), or a {"_probe_error": ...}
    stub (if the probe raised). Real results also carry _probe_ts (unix
    seconds), _probe_age_ms (set by this endpoint, how stale), and
    _probe_elapsed_ms (how long the underlying probe call took).
    """
    now_s = time.time()
    out: Dict[str, Any] = {}
    with _PROBE_LOCK:
        for name, value in _PROBE_CACHE.items():
            # Shallow copy so the caller can mutate freely.
            snapshot = dict(value)
            ts = snapshot.get("_probe_ts")
            if ts is not None:
                snapshot["_probe_age_ms"] = int((now_s - ts) * 1000)
            out[name] = snapshot
    return out


# ---------------------------------------------------------------------------
# Agents + runs
# ---------------------------------------------------------------------------

@app.get("/api/agents")
def api_agents() -> Dict[str, Any]:
    with db.session() as conn:
        agents = db.list_agents(conn)
        # decorate with run counts
        for a in agents:
            cur = conn.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done, "
                "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS err "
                "FROM run WHERE agent_id = ?", (a["id"],)
            )
            r = cur.fetchone()
            a["runs_total"] = int(r["n"] or 0)
            a["runs_done"] = int(r["done"] or 0)
            a["runs_error"] = int(r["err"] or 0)
    return {"agents": agents}


@app.get("/api/runs")
def api_runs(agent_id: Optional[int] = None, limit: int = 50,
             kind: Optional[str] = None) -> Dict[str, Any]:
    with db.session() as conn:
        runs = db.list_runs(conn, limit=min(limit, 500), agent_id=agent_id)
        # join agent name + kind so the UI can render runtime-specific badges
        # and gracefully degrade interp features for non-Qwen runs.
        agents = db.list_agents(conn)
        names = {a["id"]: a["name"] for a in agents}
        kinds = {a["id"]: (a.get("kind") or "qwen-vllm") for a in agents}
        out = []
        for r in runs:
            r["agent_name"] = names.get(r["agent_id"], "?")
            r["agent_kind"] = kinds.get(r["agent_id"], "qwen-vllm")
            if kind and r["agent_kind"] != kind:
                continue
            # peak harm score per run
            cur = conn.execute(
                "SELECT MAX(score) AS m FROM interp_score "
                "WHERE run_id = ? AND probe = 'harm'", (r["id"],)
            )
            row = cur.fetchone()
            r["harm_max"] = row["m"]
            out.append(r)
    return {"runs": out}


@app.get("/api/runs/cost_summary")
def api_runs_cost_summary(
    group_by: str = "kind", since: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate cost / tokens / yield (v1.7).

    `group_by` ∈ {'kind', 'model', 'agent'}. `since` is an optional ISO
    timestamp (e.g. '2025-01-01') to limit the window.

    NULL cost_usd values are excluded from sums (we never invent cost).
    `unknown_cost_runs` reports how many runs are missing pricing so the
    user can decide whether to add an override.
    """
    if group_by not in ("kind", "model", "agent"):
        raise HTTPException(status_code=400, detail="group_by must be kind|model|agent")
    group_col = {
        "kind":  "a.kind",
        "model": "r.model_id",
        "agent": "a.name",
    }[group_by]
    where = "r.status = 'done'"
    args: tuple = ()
    if since:
        where += " AND r.started_at >= ?"
        args = (since,)
    with db.session() as conn:
        rows = conn.execute(
            f"SELECT {group_col} AS bucket, "
            f"  COUNT(*) AS n_runs, "
            f"  SUM(COALESCE(r.tokens_in, 0))  AS tokens_in, "
            f"  SUM(COALESCE(r.tokens_out, 0)) AS tokens_out, "
            f"  SUM(r.cost_usd)                AS cost_usd, "
            f"  SUM(CASE WHEN r.cost_usd IS NULL THEN 1 ELSE 0 END) AS unknown_cost_runs, "
            f"  AVG(r.elapsed_ms)              AS avg_elapsed_ms "
            f"FROM run r LEFT JOIN agent a ON a.id = r.agent_id "
            f"WHERE {where} "
            f"GROUP BY {group_col} "
            f"ORDER BY cost_usd DESC NULLS LAST",
            args,
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) AS n_runs, "
            f"  SUM(COALESCE(r.tokens_in, 0))  AS tokens_in, "
            f"  SUM(COALESCE(r.tokens_out, 0)) AS tokens_out, "
            f"  SUM(r.cost_usd)                AS cost_usd, "
            f"  SUM(CASE WHEN r.cost_usd IS NULL THEN 1 ELSE 0 END) AS unknown_cost_runs "
            f"FROM run r LEFT JOIN agent a ON a.id = r.agent_id "
            f"WHERE {where}",
            args,
        ).fetchone()
    return {
        "group_by": group_by,
        "since":    since,
        "buckets":  [dict(r) for r in rows],
        "total":    dict(total) if total else {},
    }


@app.get("/api/pricing")
def api_pricing() -> Dict[str, Any]:
    """List known model prices (v1.7). Sourced from `agent_monitor.pricing`."""
    from agent_monitor.pricing import list_prices
    return {"prices": list_prices()}


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: int) -> Dict[str, Any]:
    with db.session() as conn:
        cur = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="run not found")
        run = dict(row)
        cur = conn.execute("SELECT name, kind FROM agent WHERE id = ?", (run["agent_id"],))
        ag = cur.fetchone()
        run["agent_name"] = ag["name"] if ag else "?"
        run["agent_kind"] = (ag["kind"] if ag else None) or "qwen-vllm"
        traces = db.list_trace(conn, run_id)
        scores = db.list_interp_scores(conn, run_id)
        nla = db.list_nla_decodings(conn, run_id)
        signals = db.list_classifier_signals(conn, run_id)
    return {
        "run": run, "trace": traces, "interp": scores, "nla": nla,
        "classifier_signals": signals,
    }


# ---------------------------------------------------------------------------
# v1.8: defender-side classifier endpoints
# ---------------------------------------------------------------------------

@app.get("/api/classifier/signatures")
def api_classifier_signatures() -> Dict[str, Any]:
    """List the active offensive_patterns signatures with weights + sources.

    The UI renders this as a transparency surface: every match the
    classifier reports can be traced back to a public source URL.
    """
    try:
        from agent_monitor.classifiers.offensive_patterns import list_signatures
    except Exception as e:
        raise HTTPException(status_code=503,
            detail=f"classifier feature not available: {e}")
    return {"classifier": "offensive_patterns", "signatures": list_signatures()}


@app.get("/api/classifier/posture")
def api_classifier_posture(
    min_score: float = 0.05, limit: int = 100,
) -> Dict[str, Any]:
    """Return runs sorted by classifier_score desc.

    `min_score` filters out quiet runs. The UI's Posture tab is the
    intended consumer; researchers reviewing their fleet see exactly
    which agent runs exhibit exploit-dev patterns and why.
    """
    with db.session() as conn:
        rows = conn.execute(
            "SELECT r.id, r.agent_id, r.external_id, r.status, r.started_at, "
            "  r.classifier_score, r.classifier_kind, "
            "  a.name AS agent_name, a.kind AS agent_kind "
            "FROM run r LEFT JOIN agent a ON a.id = r.agent_id "
            "WHERE r.classifier_score IS NOT NULL "
            "  AND r.classifier_score >= ? "
            "ORDER BY r.classifier_score DESC, r.started_at DESC "
            "LIMIT ?",
            (float(min_score), int(max(1, min(limit, 1000)))),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["signals"] = db.list_classifier_signals(conn, d["id"])
            out.append(d)
    return {"runs": out, "min_score": min_score}


class _ReplayBody(BaseModel):
    since:    Optional[str] = None
    limit:    int = 1000
    only_null: bool = False    # if True, only re-classify rows where score is NULL


@app.post("/api/classifier/replay")
def api_classifier_replay(body: _ReplayBody) -> Dict[str, Any]:
    """Re-run `offensive_patterns` over historical runs.

    Useful after upgrading the signature library or when ingesting an
    older DB. Returns counts of {classified, scored, skipped}.

    `only_null=True` skips runs that already have a score (cheaper).
    """
    try:
        from agent_monitor.classifiers.offensive_patterns import classify_run
    except Exception as e:
        raise HTTPException(status_code=503,
            detail=f"classifier feature not available: {e}")
    where = ["1=1"]
    args: List[Any] = []
    if body.since:
        where.append("r.started_at >= ?")
        args.append(body.since)
    if body.only_null:
        where.append("r.classifier_score IS NULL")
    sql = (
        "SELECT id FROM run r WHERE " + " AND ".join(where) +
        " ORDER BY started_at DESC LIMIT ?"
    )
    args.append(int(max(1, min(body.limit, 10_000))))
    classified = scored = 0
    with db.session() as conn:
        ids = [int(r["id"]) for r in conn.execute(sql, args).fetchall()]
    for rid in ids:
        try:
            res = classify_run(rid)
            with db.session() as conn:
                db.persist_classifier_result(
                    conn, rid, classifier="offensive_patterns",
                    score=res["score"], kind=res["kind"],
                    signals=res["signals"],
                )
            classified += 1
            if res["score"] and res["score"] > 0:
                scored += 1
        except Exception:
            continue
    return {
        "classified": classified, "scored_nonzero": scored,
        "skipped": len(ids) - classified,
    }


@app.post("/api/runs/{run_id}/classify")
def api_run_classify(run_id: int) -> Dict[str, Any]:
    """Run the offensive_patterns classifier on a single run id.

    Convenience for "I just changed signatures, re-classify this one
    run." Idempotent.
    """
    from agent_monitor.classifiers.offensive_patterns import classify_run
    with db.session() as conn:
        if not conn.execute("SELECT 1 FROM run WHERE id = ?", (run_id,)).fetchone():
            raise HTTPException(status_code=404, detail="run not found")
    res = classify_run(run_id)
    with db.session() as conn:
        db.persist_classifier_result(
            conn, run_id, classifier="offensive_patterns",
            score=res["score"], kind=res["kind"], signals=res["signals"],
        )
    return res


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

@app.get("/api/memory")
def api_memory(q: Optional[str] = None, semantic: bool = False, limit: int = 30) -> Dict[str, Any]:
    _require(memory, "long-term memory")
    if not q:
        return {"results": memory.list_recent(limit=limit), "mode": "recent"}
    if semantic:
        sem = memory.search_semantic(q, limit=limit)
        return {
            "results": [{"score": s, **chunk} for chunk, s in sem],
            "mode": "semantic",
        }
    return {"results": memory.search_text(q, limit=limit), "mode": "text"}


@app.post("/api/memory")
def api_memory_add(body: MemoryIn) -> Dict[str, Any]:
    _require(memory, "long-term memory")
    rid = memory.remember(
        body.text, source=body.source, kind=body.kind,
        tags=body.tags, with_vector=body.with_vector,
    )
    return {"id": rid}


# ---------------------------------------------------------------------------
# Interp
# ---------------------------------------------------------------------------

@app.get("/api/interp/status")
def api_interp_status() -> Dict[str, Any]:
    _require(interp_bridge, "interp probes", "ml")
    out = interp_bridge.status()
    try:
        from agent_monitor import safety_llamaguard
        out["llama_guard"] = safety_llamaguard.status()
    except Exception as e:
        out["llama_guard"] = {"ready": False, "error": str(e)}
    return out


@app.post("/api/interp/score")
def api_interp_score(body: ScoreIn) -> Dict[str, Any]:
    _require(interp_bridge, "interp probes", "ml")
    # Either toy probes or Llama Guard 3 is enough to satisfy a score request.
    toy_ready = interp_bridge.is_ready()
    lg_ready = False
    try:
        from agent_monitor import safety_llamaguard
        lg_ready = safety_llamaguard.is_ready()
    except Exception:
        lg_ready = False
    if not (toy_ready or lg_ready):
        raise HTTPException(status_code=503, detail="no safety classifier available")
    return {"text_chars": len(body.text), "scores": interp_bridge.score_all(body.text)}


# ---------------------------------------------------------------------------
# NLA -- Natural Language Autoencoders
# ---------------------------------------------------------------------------

@app.get("/api/nla/status")
def api_nla_status() -> Dict[str, Any]:
    _require(nla_client, "NLA pipeline")
    s = nla_client.status()
    if nla_worker is not None:
        s["worker"] = nla_worker.stats()
    return s


@app.get("/api/nla/cache")
def api_nla_cache() -> Dict[str, Any]:
    _require(nla_cache, "NLA pipeline")
    return nla_cache.stats()


@app.post("/api/nla/cache/clear")
def api_nla_cache_clear() -> Dict[str, Any]:
    _require(nla_cache, "NLA pipeline")
    return nla_cache.clear()


@app.get("/api/nla/queue")
def api_nla_queue() -> Dict[str, Any]:
    _require(nla_worker, "NLA pipeline")
    return nla_worker.stats()


@app.post("/api/nla/local/enable")
def api_nla_local_enable() -> Dict[str, Any]:
    """Trigger an explicit load of the local_activations backend (Qwen 2.5).

    The model is multi-GB and we never auto-load it. Hitting this endpoint
    once per process is the explicit opt-in.
    """
    try:
        from agent_monitor import transformers_runtime
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"transformers_runtime unavailable: {e}. "
                "Install with: pip install 'cogniguardai[ml]'"
            ),
        )
    err = transformers_runtime.ensure_loaded()
    return {
        "ok": err is None,
        "error": err,
        "status": transformers_runtime.status(),
    }


@app.post("/api/nla/decode")
def api_nla_decode(body: NLADecodeIn) -> Dict[str, Any]:
    _require(nla_client, "NLA pipeline")
    if not nla_client.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "no NLA backend available -- set NLA_REMOTE_URL or pull "
                f"'{nla_client.PROMPTED_MODEL}' into Ollama"
            ),
        )
    decoding = nla_client.decode(
        body.text,
        target_model=body.target_model,
        layer=body.layer,
    )
    persisted_id: Optional[int] = None
    if body.persist and decoding.get("ok"):
        try:
            with db.session() as conn:
                persisted_id = db.record_nla_decoding(
                    conn,
                    run_id=body.run_id,
                    target=body.target,
                    decoding=decoding,
                )
        except Exception:
            # never fail a decode because of a persistence hiccup
            persisted_id = None
    return {"decoding": decoding, "persisted_id": persisted_id}


# ---------------------------------------------------------------------------
# Code scanning (v1.5) -- LLM-driven source-code screening.
#
# This is INTENTIONALLY narrower than the marketing for "AI security
# scanners". See PRODUCTION_NLA_PLAN.md for the "what this can/cannot do"
# section. The endpoints here are deliberately small: start, status, list,
# findings, cancel, decode-one-snippet.
# ---------------------------------------------------------------------------

class CodeScanStartIn(BaseModel):
    root_path: str
    label: Optional[str] = None
    # everything below is optional; sensible defaults live in code_scan.py
    max_bytes: Optional[int] = None
    max_chunk_lines: Optional[int] = None
    overlap_lines: Optional[int] = None
    max_files: Optional[int] = None
    persist_low: Optional[bool] = None
    extensions: Optional[Dict[str, str]] = None
    skip_dirs: Optional[list[str]] = None
    # When set, only scan files git knows have changed since this ref
    # (plus untracked-not-ignored). Fails loudly if the path isn't a
    # git repo. Examples: "HEAD~1", "main", "abc123".
    git_since: Optional[str] = None


class CodeDecodeIn(BaseModel):
    code: str
    language: Optional[str] = None
    path_hint: Optional[str] = None


@app.get("/api/scan/status")
def api_scan_status() -> Dict[str, Any]:
    """Module-level status: model, prompt version, axes, honest caveat."""
    _require(nla_client, "code scanning")
    return nla_client.code_status()


@app.post("/api/scan/start")
def api_scan_start(body: CodeScanStartIn) -> Dict[str, Any]:
    _require(code_scan_mod, "code scanning")
    opts: Dict[str, Any] = {}
    for k in ("max_bytes", "max_chunk_lines", "overlap_lines",
              "max_files", "persist_low", "extensions", "git_since"):
        v = getattr(body, k)
        if v is not None:
            opts[k] = v
    if body.skip_dirs is not None:
        opts["skip_dirs"] = set(body.skip_dirs)
    result = code_scan_mod.start_scan(
        body.root_path, label=body.label, options=opts,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


class _ExternalFinding(BaseModel):
    file_path:    str
    kind:         str = "external"
    severity:     str = "info"
    line:         Optional[int] = None
    end_line:     Optional[int] = None
    excerpt:      str = ""
    message:      str = ""
    rule_id:      Optional[str] = None
    language:     Optional[str] = None
    file_sha256:  Optional[str] = None


class _ExternalScanIngest(BaseModel):
    tool_name:  str
    root_path:  str
    label:      Optional[str] = None
    findings:   List[_ExternalFinding]
    # v1.9 scanner-observability fields. All optional. None = unknown.
    cost_usd:        Optional[float] = None
    ci_minutes:      Optional[float] = None
    scanner_version: Optional[str]   = None
    elapsed_ms:      Optional[int]   = None


@app.post("/api/scan/external")
def api_scan_external(body: _ExternalScanIngest) -> Dict[str, Any]:
    """Ingest findings from an external static-analysis tool (Semgrep,
    CodeQL, Bandit, custom linter). The findings appear in the Code
    Scan tab alongside built-in scanner output. v1.7+

    v1.9 extension: caller can report `cost_usd`, `ci_minutes`,
    `scanner_version`, `elapsed_ms`. These populate the Scanner Obs
    dashboard ($/finding, version drift, etc). Unknown = NULL.

    AgentMonitor does NOT run the tool, write rules, or interpret what
    each finding means -- we just persist whatever the caller reports.
    """
    try:
        from agent_monitor.adapters.findings import ingest_findings
    except Exception as e:
        raise HTTPException(status_code=503,
            detail=f"external-findings ingestion not available: {e}")
    payload = [f.model_dump() for f in body.findings]
    return ingest_findings(
        tool_name=body.tool_name, root_path=body.root_path,
        label=body.label, findings=payload,
        cost_usd=body.cost_usd, ci_minutes=body.ci_minutes,
        scanner_version=body.scanner_version, elapsed_ms=body.elapsed_ms,
    )


class _SarifIngest(BaseModel):
    """Body for POST /api/scan/external/sarif. The `sarif` field is the
    SARIF v2.1.0 document itself (the JSON the tool produced)."""
    root_path:       str
    sarif:           Dict[str, Any]
    label:           Optional[str] = None
    tool_override:   Optional[str] = None
    # v1.9 scanner-observability fields. All optional.
    cost_usd:        Optional[float] = None
    ci_minutes:      Optional[float] = None
    elapsed_ms:      Optional[int]   = None


@app.post("/api/scan/external/sarif")
def api_scan_external_sarif(body: _SarifIngest) -> Dict[str, Any]:
    """Ingest a SARIF v2.1.0 document. Any SARIF-emitting tool
    (Semgrep, CodeQL, Bandit, Snyk, Trivy, ESLint w/ SARIF formatter,
    Checkov, Gitleaks, ...) flows through here with no per-tool code.

    Tool name and scanner version are auto-detected from
    `runs[].tool.driver.{name, semanticVersion|version}`. One SARIF
    document can contain multiple runs; each becomes its own code_scan
    row in the Code Scan tab.

    AgentMonitor does NOT run the tool, write rules, or interpret the
    findings -- we just persist whatever the SARIF says, normalized to
    our schema so all scanners share one pane of glass.
    """
    try:
        from agent_monitor.adapters.findings import ingest_sarif
    except Exception as e:
        raise HTTPException(status_code=503,
            detail=f"SARIF ingestion not available: {e}")
    try:
        return ingest_sarif(
            body.sarif,
            root_path=body.root_path,
            label=body.label,
            tool_override=body.tool_override,
            cost_usd=body.cost_usd,
            ci_minutes=body.ci_minutes,
            elapsed_ms=body.elapsed_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"SARIF parse error: {e}")


class _SandboxReportIngest(BaseModel):
    """Body for POST /api/scan/external/sandbox. The `report` field is
    the sandbox JSON itself (Cuckoo 2.x or our generic envelope).

    Use ``format`` to force a parser ("cuckoo" / "generic"); leave it
    None for auto-detection. ``root_path`` should be a stable batch
    label (e.g. ``"sandbox/prod"``) -- drift detection groups by
    ``tool + root_path``, so a per-run path defeats drift entirely.
    """
    root_path:       str
    report:          Dict[str, Any]
    label:           Optional[str] = None
    format:          Optional[str] = None
    tool_override:   Optional[str] = None
    # v1.9 scanner-observability fields. All optional.
    cost_usd:        Optional[float] = None
    ci_minutes:      Optional[float] = None
    elapsed_ms:      Optional[int]   = None


@app.post("/api/scan/external/sandbox")
def api_scan_external_sandbox(body: _SandboxReportIngest) -> Dict[str, Any]:
    """Ingest a VM/sandbox JSON report (Cuckoo 2.x or generic envelope).

    This is the *dynamic*-analysis sister to
    ``/api/scan/external/sarif``: where SARIF surfaces what a static
    analyzer thought of source code, this surfaces what a sandbox saw
    when it actually detonated a sample. Each signature / signal
    becomes one `code_finding`, the sample's SHA-256 becomes
    `file_path`, and the existing Code Scan + Scanner Obs UIs work
    without changes.

    AgentMonitor does NOT run the sandbox, score behaviors, or decide
    whether something is malicious -- we persist exactly what the
    report said, mapped to our uniform schema.
    """
    try:
        from agent_monitor.adapters.findings import ingest_sandbox_report
    except Exception as e:
        raise HTTPException(status_code=503,
            detail=f"sandbox-report ingestion not available: {e}")
    try:
        return ingest_sandbox_report(
            body.report,
            root_path=body.root_path,
            label=body.label,
            format=body.format,
            tool_override=body.tool_override,
            cost_usd=body.cost_usd,
            ci_minutes=body.ci_minutes,
            elapsed_ms=body.elapsed_ms,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"sandbox report parse error: {e}")


@app.get("/api/sandbox/detonations")
def api_sandbox_detonations(limit: int = 100) -> Dict[str, Any]:
    """List recent VM/sandbox detonations (v1.10).

    Returns the scans that came in via ``/api/scan/external/sandbox``,
    each with its sample identifier and per-severity signature count.
    This is the data backing the *Detonations* UI panel; the same rows
    are also visible (mixed with static-analysis scans) under
    ``/api/scan/list``.
    """
    with db.session() as conn:
        rows = db.list_sandbox_detonations(conn, limit=max(1, min(limit, 1000)))
    return {"detonations": rows, "count": len(rows)}


# --- v1.9: triage + Scanner Obs endpoints ----------------------------------

class _TriageBody(BaseModel):
    state: str
    note:  Optional[str] = None
    by:    Optional[str] = None


@app.post("/api/findings/{finding_id}/triage")
def api_finding_triage(finding_id: int, body: _TriageBody) -> Dict[str, Any]:
    """Set/update a finding's triage state.

    `state` is one of: new | confirmed | false_positive | fixed |
    wontfix | suppressed. Setting 'fixed' stamps fixed_at = now.
    """
    try:
        with db.session() as conn:
            return db.triage_finding(
                conn, finding_id, state=body.state, note=body.note, by=body.by,
            )
    except KeyError:
        raise HTTPException(status_code=404, detail="finding not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/scanner_obs/summary")
def api_scanner_obs_summary(since: Optional[str] = None) -> Dict[str, Any]:
    """Fleet-wide scanner KPIs: $/finding, FP rate, time-to-fix p50/p90."""
    with db.session() as conn:
        return db.scanner_obs_summary(conn, since=since)


@app.get("/api/scanner_obs/tools")
def api_scanner_obs_tools(since: Optional[str] = None) -> Dict[str, Any]:
    """Per-tool breakdown -- one row per distinct scanner."""
    with db.session() as conn:
        return {"tools": db.scanner_obs_per_tool(conn, since=since)}


@app.get("/api/scanner_obs/drift")
def api_scanner_obs_drift(
    tool: str, root_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare the two most-recent scans of (tool, root_path). Returns
    new / persistent / gone fingerprints + their finding metadata."""
    with db.session() as conn:
        return db.scanner_obs_drift(conn, tool=tool, root_path=root_path)


@app.get("/api/scanner_obs/density")
def api_scanner_obs_density(
    since: Optional[str] = None, top_n: int = 15,
) -> Dict[str, Any]:
    """Findings density per `kind` with triage breakdown."""
    with db.session() as conn:
        return {"kinds": db.scanner_obs_density(conn, since=since, top_n=top_n)}


@app.get("/api/scan/list")
def api_scan_list(limit: int = 50) -> Dict[str, Any]:
    with db.session() as conn:
        scans = db.list_code_scans(conn, limit=limit)
    return {"scans": scans}


@app.get("/api/scan/{scan_id}")
def api_scan_detail(scan_id: int) -> Dict[str, Any]:
    with db.session() as conn:
        scan = db.get_code_scan(conn, scan_id)
        if not scan:
            raise HTTPException(status_code=404, detail="scan not found")
        hist = db.code_scan_severity_histogram(conn, scan_id)
    runtime = (code_scan_mod.runtime_status(scan_id)
               if code_scan_mod is not None else {"available": False})
    return {
        "scan": scan,
        "histogram": hist,
        "runtime": runtime,
    }


@app.get("/api/scan/{scan_id}/findings")
def api_scan_findings(
    scan_id: int, min_severity: Optional[str] = None,
    kind: Optional[str] = None, file_path: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    with db.session() as conn:
        scan = db.get_code_scan(conn, scan_id)
        if not scan:
            raise HTTPException(status_code=404, detail="scan not found")
        findings = db.list_code_findings(
            conn, scan_id,
            min_severity=min_severity, kind=kind, file_path=file_path,
            limit=max(1, min(limit, 5000)),
        )
    return {"scan_id": scan_id, "count": len(findings), "findings": findings}


@app.post("/api/scan/{scan_id}/cancel")
def api_scan_cancel(scan_id: int) -> Dict[str, Any]:
    _require(code_scan_mod, "code scanning")
    ok = code_scan_mod.cancel(scan_id)
    if not ok:
        raise HTTPException(
            status_code=404,
            detail="scan not active (already done or never started)",
        )
    return {"ok": True, "cancel_requested": True}


@app.post("/api/scan/decode")
def api_scan_decode(body: CodeDecodeIn) -> Dict[str, Any]:
    """One-shot decode: handy for testing the prompt without a full scan."""
    _require(nla_client, "code scanning")
    result = nla_client.decode_code(
        body.code, language=body.language, path_hint=body.path_hint,
    )
    return {"decoding": result}


# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------

@app.get("/api/browser/status")
def api_browser_status() -> Dict[str, Any]:
    _require(browser_mod, "browser automation", "browser")
    s = browser_mod._SESSION
    return {
        "open": s is not None and s.is_open,
        "last_url": getattr(s, "last_url", None),
        "last_title": getattr(s, "last_title", None),
    }


@app.post("/api/browser/goto")
def api_browser_goto(body: GotoIn) -> Dict[str, Any]:
    _require(browser_mod, "browser automation", "browser")
    sess = browser_mod.get_or_start(headless=body.headless)
    return sess.goto(body.url)


@app.post("/api/browser/close")
def api_browser_close() -> Dict[str, Any]:
    _require(browser_mod, "browser automation", "browser")
    browser_mod.shutdown()
    return {"ok": True}


@app.get("/api/browser/screenshot.png")
def api_browser_screenshot(full_page: bool = False):
    _require(browser_mod, "browser automation", "browser")
    s = browser_mod._SESSION
    if s is None or not s.is_open:
        raise HTTPException(status_code=409, detail="no browser session open")
    return Response(content=s.screenshot(full_page=full_page), media_type="image/png")


# ---------------------------------------------------------------------------
# Live trace WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/runs/{run_id}")
async def ws_run_trace(ws: WebSocket, run_id: int):
    await ws.accept()
    last_seq = -1
    try:
        # send initial backfill
        with db.session() as conn:
            traces = db.list_trace(conn, run_id)
        for t in traces:
            await ws.send_json({"type": "trace", **t})
            last_seq = max(last_seq, int(t["seq"]))
        # stream new events by polling (cheap with WAL)
        while True:
            await asyncio.sleep(0.6)
            with db.session() as conn:
                rows = conn.execute(
                    "SELECT * FROM trace_event WHERE run_id = ? AND seq > ? ORDER BY seq",
                    (run_id, last_seq),
                ).fetchall()
                cur = conn.execute("SELECT status FROM run WHERE id = ?", (run_id,))
                row = cur.fetchone()
                status = row["status"] if row else None
            for r in rows:
                d = dict(r)
                await ws.send_json({"type": "trace", **d})
                last_seq = max(last_seq, int(d["seq"]))
            if status in ("done", "error"):
                await ws.send_json({"type": "status", "status": status})
                break
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def ui_root():
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse(
            "<h1>AgentMonitor</h1><p>UI not built yet.</p>", status_code=200,
        )
    return FileResponse(idx)
