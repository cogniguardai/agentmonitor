"""
agent_monitor.db — SQLite schema + thin sync CRUD helpers.

Why SQLite (beginner note):
    A single file, zero config, durable, ACID, perfect for a desktop app.
    Same database engine Firefox / iMessage / WhatsApp use locally.

Why sync (not asyncio):
    SQLite is fast enough on a single thread that the agent runner can
    be plain sync. The FastAPI layer (Phase B) will use aiosqlite for
    request handlers; both speak the same .db file.

Schema overview:

    agent            one row per agent class (customer_support, sop_processor, ...)
    run              one row per execution; FK to agent
    trace_event      time-series events emitted during a run; FK to run
    memory_chunk     long-term retrievable text (with optional embedding blob)
    interp_score     stored harm/refusal/hedging probabilities; FK to run

All rows have created_at = ISO-8601 UTC timestamp.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_monitor import DB_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agent (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    kind            TEXT NOT NULL DEFAULT 'qwen-vllm',  -- runtime: qwen-vllm | openai | anthropic | ollama | langchain | autogen | smolagents | custom
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        INTEGER NOT NULL REFERENCES agent(id) ON DELETE CASCADE,
    external_id     TEXT,                   -- e.g. ticket id "T-001"
    status          TEXT NOT NULL,          -- 'pending' | 'running' | 'done' | 'error'
    input_text      TEXT,
    output_text     TEXT,
    meta_json       TEXT,                   -- arbitrary metadata (kairos loops, etc.)
    elapsed_ms      INTEGER,
    -- v1.7: economics. NULL when we don't know (unknown model, unreported tokens).
    -- We never *guess* cost; if pricing.compute_cost() returns None we store NULL,
    -- and the UI shows '—' instead of $0 (which would be a lie).
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    model_id        TEXT,                   -- e.g. 'gpt-4o-mini', 'qwen2.5-7b'
    cost_usd        REAL,
    -- v1.8: defender-side trace classifier (offensive_patterns).
    -- NULL = not yet classified OR no signals (below NULL_BELOW threshold).
    -- See `agent_monitor.classifiers.offensive_patterns`.
    classifier_score REAL,
    classifier_kind  TEXT,                  -- dominant domain: 're_tooling' | 'kernel_api' | 'byovd' | 'exploit_lexicon' | 'attack_technique' | 'recon' | NULL
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);

CREATE TABLE IF NOT EXISTS classifier_signal (
    -- One row per signature hit on a given run. Lets the UI explain
    -- WHY a score was assigned without storing a giant JSON blob.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    classifier      TEXT NOT NULL,          -- e.g. 'offensive_patterns'
    signature_id    TEXT NOT NULL,          -- e.g. 'ke.api.MmMapIoSpace'
    domain          TEXT NOT NULL,          -- e.g. 'kernel_api'
    weight          REAL NOT NULL,
    matched_text    TEXT,                   -- truncated excerpt from the trace
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS classifier_signal_run_idx
    ON classifier_signal(run_id);
CREATE INDEX IF NOT EXISTS run_agent_idx       ON run(agent_id);
CREATE INDEX IF NOT EXISTS run_status_idx      ON run(status);
CREATE INDEX IF NOT EXISTS run_started_at_idx  ON run(started_at);

CREATE TABLE IF NOT EXISTS trace_event (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,        -- monotonic per-run counter
    kind            TEXT NOT NULL,           -- 'model_call' | 'kairos' | 'interp' | 'tool' | 'log'
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS trace_run_seq_idx   ON trace_event(run_id, seq);

CREATE TABLE IF NOT EXISTS memory_chunk (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,           -- e.g. 'run:42' or 'manual'
    kind            TEXT NOT NULL,           -- 'input' | 'output' | 'note'
    text            TEXT NOT NULL,
    embedding_blob  BLOB,                    -- optional float32 vector
    embed_dim       INTEGER,
    tags            TEXT,                    -- comma-separated
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_kind_idx     ON memory_chunk(kind);
CREATE INDEX IF NOT EXISTS memory_source_idx   ON memory_chunk(source);

CREATE TABLE IF NOT EXISTS interp_score (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    target          TEXT NOT NULL,           -- 'input' | 'output' | 'cot'
    probe           TEXT NOT NULL,           -- 'harm' | 'refusal' | 'hedging'
    score           REAL NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS interp_run_idx      ON interp_score(run_id);

CREATE TABLE IF NOT EXISTS nla_decoding (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES run(id) ON DELETE CASCADE,
    trace_seq       INTEGER,                 -- nullable: ad-hoc decodings have no trace event
    target          TEXT NOT NULL,           -- 'input' | 'output' | 'cot' | 'adhoc'
    backend         TEXT NOT NULL,           -- 'remote' | 'prompted_approx'
    model           TEXT,                    -- e.g. 'kitft/nla-qwen2.5-7b-L20-av' or 'qwen2.5-coder:3b'
    explanation     TEXT,                    -- single-line summary used for UI
    topic           TEXT,
    evaluation_awareness REAL,                -- nullable; 0..1 when prompted_approx
    hidden_motivation    REAL,
    safety_relevance     REAL,
    notes_json      TEXT,                    -- JSON-encoded list of evidence strings
    raw_json        TEXT,                    -- raw backend response, for audit
    latency_ms      INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS nla_run_idx        ON nla_decoding(run_id);
CREATE INDEX IF NOT EXISTS nla_run_target_idx ON nla_decoding(run_id, target);

-- v1.5: code scanning ---------------------------------------------------
CREATE TABLE IF NOT EXISTS code_scan (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path       TEXT NOT NULL,
    label           TEXT,                       -- user-supplied (e.g. 'linux-6.6 fs/')
    status          TEXT NOT NULL,              -- 'queued' | 'running' | 'done' | 'error' | 'cancelled'
    model           TEXT,                       -- model id used (e.g. 'qwen2.5-coder:3b' OR 'external:semgrep')
    prompt_version  TEXT,                       -- e.g. 'v1.5.0'
    options_json    TEXT,                       -- JSON of extensions, max_bytes, max_chunk_lines, etc.
    total_files     INTEGER DEFAULT 0,
    scanned_files   INTEGER DEFAULT 0,
    skipped_files   INTEGER DEFAULT 0,
    findings_count  INTEGER DEFAULT 0,
    error           TEXT,
    -- v1.9: scanner-observability layer.
    -- The caller reports cost and scanner version on ingest.
    -- NULL means "unknown" -- we never invent a number.
    cost_usd        REAL,
    ci_minutes      REAL,
    scanner_version TEXT,                       -- e.g. 'semgrep 1.45.0', 'codeql 2.16.0'
    elapsed_ms      INTEGER,                    -- wall-clock duration of the external run
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS code_scan_status_idx ON code_scan(status);

CREATE TABLE IF NOT EXISTS code_finding (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id         INTEGER NOT NULL REFERENCES code_scan(id) ON DELETE CASCADE,
    file_path       TEXT NOT NULL,              -- path relative to scan.root_path
    file_sha256     TEXT,                       -- hash of file content (dedupe across rescans)
    chunk_index     INTEGER NOT NULL,           -- 0-based chunk within file
    chunk_start_line INTEGER NOT NULL,          -- 1-indexed first line of chunk
    chunk_end_line  INTEGER NOT NULL,           -- 1-indexed last line of chunk (inclusive)
    language        TEXT,
    kind            TEXT NOT NULL,              -- one of CODE_RISK_AXES
    severity        TEXT NOT NULL,              -- 'info'|'low'|'medium'|'high'|'critical'
    severity_rank   INTEGER NOT NULL,           -- 0..4 for ORDER BY DESC
    line_hint       INTEGER,                    -- line within the chunk, 1-indexed
    excerpt         TEXT NOT NULL,              -- verbatim quote from the source
    explanation     TEXT,
    chunk_summary   TEXT,                       -- the model's one-liner about this chunk
    created_at      TEXT NOT NULL,
    -- v1.9: scanner-observability layer ----------------------------------
    -- A stable hash of (scan.model, file_path, kind, rule_id, normalised
    -- excerpt). Same fingerprint across scans = same finding -- this is
    -- how drift detection works. NULL on rows ingested pre-v1.9.
    fingerprint     TEXT,
    -- Triage workflow. Defaults to 'new' on insert. The UI can flip it.
    -- 'confirmed'      -- human reviewed, real finding
    -- 'false_positive' -- human reviewed, not actually a problem
    -- 'fixed'          -- the underlying code was fixed
    -- 'wontfix'        -- accepted-risk
    -- 'suppressed'     -- noise we don't want to see again
    triage_state    TEXT,
    triage_note     TEXT,
    triage_by       TEXT,                       -- free-form actor name
    triage_at       TEXT,
    fixed_at        TEXT                        -- set when triage_state := 'fixed'
);
CREATE INDEX IF NOT EXISTS code_finding_scan_idx   ON code_finding(scan_id);
CREATE INDEX IF NOT EXISTS code_finding_sev_idx    ON code_finding(scan_id, severity_rank DESC);
CREATE INDEX IF NOT EXISTS code_finding_file_idx   ON code_finding(scan_id, file_path);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path = DB_PATH):
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    """Create tables if they don't exist. Safe to call repeatedly.

    Also runs idempotent ALTER-TABLE migrations for columns added after the
    initial schema (so older DBs upgrade in place without losing data).
    """
    with session(path) as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate_v1_6_agent_kind(conn)
        _migrate_v1_7_run_economics(conn)
        _migrate_v1_8_classifier(conn)
        _migrate_v1_9_scanner_obs(conn)


def _migrate_v1_6_agent_kind(conn) -> None:
    """v1.6: add `agent.kind` so the dashboard can honestly report which
    runtime produced each run (Qwen vLLM, OpenAI, Anthropic, Ollama, ...).

    Existing rows are stamped with 'qwen-vllm' since that was the only
    runtime before v1.6. New runs from non-Qwen adapters will set their
    own value via `upsert_agent(..., kind=...)`.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(agent)").fetchall()]
    if "kind" not in cols:
        conn.execute("ALTER TABLE agent ADD COLUMN kind TEXT NOT NULL DEFAULT 'qwen-vllm'")


def _migrate_v1_7_run_economics(conn) -> None:
    """v1.7: per-run token / cost accounting.

    Pre-v1.7 rows have NULLs which the UI renders as '—'. We do not
    backfill cost retroactively because we don't know which model was
    used; lying about historical cost would defeat the point of the
    feature.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(run)").fetchall()]
    for col, ddl in (
        ("tokens_in",  "ALTER TABLE run ADD COLUMN tokens_in INTEGER"),
        ("tokens_out", "ALTER TABLE run ADD COLUMN tokens_out INTEGER"),
        ("model_id",   "ALTER TABLE run ADD COLUMN model_id TEXT"),
        ("cost_usd",   "ALTER TABLE run ADD COLUMN cost_usd REAL"),
    ):
        if col not in cols:
            conn.execute(ddl)


def _migrate_v1_8_classifier(conn) -> None:
    """v1.8: defender-side trace classifier columns + signal table.

    The `classifier_signal` table is created by `executescript` from
    SCHEMA_SQL above; this function only handles the column additions
    on the existing `run` table for in-place upgrades.

    Pre-v1.8 rows have NULL classifier_score / classifier_kind. The
    user can backfill via `POST /api/classifier/replay`.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(run)").fetchall()]
    for col, ddl in (
        ("classifier_score", "ALTER TABLE run ADD COLUMN classifier_score REAL"),
        ("classifier_kind",  "ALTER TABLE run ADD COLUMN classifier_kind  TEXT"),
    ):
        if col not in cols:
            conn.execute(ddl)


def persist_classifier_result(
    conn, run_id: int, *, classifier: str, score: float,
    kind: Optional[str], signals: Iterable[Dict[str, Any]],
) -> None:
    """Persist a classifier result for a run.

    Replaces any prior signals from the same `classifier` so re-running
    the classifier on a run is idempotent.
    """
    conn.execute(
        "DELETE FROM classifier_signal WHERE run_id = ? AND classifier = ?",
        (run_id, classifier),
    )
    for s in signals:
        conn.execute(
            "INSERT INTO classifier_signal (run_id, classifier, signature_id, "
            "domain, weight, matched_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, classifier, s["id"], s["domain"], float(s["weight"]),
             s.get("matched_text", "")[:500], _utcnow()),
        )
    # Only persist score/kind on the run row when there's actual signal.
    # NULL means "classified but quiet"; we use a tiny epsilon to keep
    # signal rows recoverable but not pollute the dashboard.
    score_val: Optional[float] = float(score) if score and score > 0 else None
    conn.execute(
        "UPDATE run SET classifier_score = ?, classifier_kind = ? WHERE id = ?",
        (score_val, kind, run_id),
    )


def list_classifier_signals(conn, run_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM classifier_signal WHERE run_id = ? "
        "ORDER BY weight DESC, signature_id", (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# v1.9: scanner-observability layer
# ---------------------------------------------------------------------------
#
# AgentMonitor as the META-tool over Semgrep / CodeQL / Bandit / your own
# scanner. The user runs their tool, ingests findings via item-4 plumbing
# (`POST /api/scan/external`), and AgentMonitor tracks:
#
#   * $/finding over time      -- cost / n_findings, by tool
#   * False-positive rate      -- triage_state='false_positive' / triaged
#   * Time-to-fix (TTF)        -- fixed_at - created_at, p50 / p90
#   * Finding density by kind  -- count grouped by `kind`
#   * Drift between scans      -- new / persistent / gone fingerprints
#
# None of these require AgentMonitor to BE a scanner. We are the dashboard.
# ---------------------------------------------------------------------------

def _migrate_v1_9_scanner_obs(conn) -> None:
    """v1.9: scanner-observability columns on code_scan + code_finding."""
    cs_cols = [r[1] for r in conn.execute("PRAGMA table_info(code_scan)").fetchall()]
    for col, ddl in (
        ("cost_usd",        "ALTER TABLE code_scan ADD COLUMN cost_usd REAL"),
        ("ci_minutes",      "ALTER TABLE code_scan ADD COLUMN ci_minutes REAL"),
        ("scanner_version", "ALTER TABLE code_scan ADD COLUMN scanner_version TEXT"),
        ("elapsed_ms",      "ALTER TABLE code_scan ADD COLUMN elapsed_ms INTEGER"),
    ):
        if col not in cs_cols:
            conn.execute(ddl)

    cf_cols = [r[1] for r in conn.execute("PRAGMA table_info(code_finding)").fetchall()]
    for col, ddl in (
        ("fingerprint",  "ALTER TABLE code_finding ADD COLUMN fingerprint TEXT"),
        ("triage_state", "ALTER TABLE code_finding ADD COLUMN triage_state TEXT"),
        ("triage_note",  "ALTER TABLE code_finding ADD COLUMN triage_note TEXT"),
        ("triage_by",    "ALTER TABLE code_finding ADD COLUMN triage_by TEXT"),
        ("triage_at",    "ALTER TABLE code_finding ADD COLUMN triage_at TEXT"),
        ("fixed_at",     "ALTER TABLE code_finding ADD COLUMN fixed_at TEXT"),
    ):
        if col not in cf_cols:
            conn.execute(ddl)
    # Backfill triage_state='new' for any row that has NULL (older inserts).
    conn.execute(
        "UPDATE code_finding SET triage_state = 'new' "
        "WHERE triage_state IS NULL OR triage_state = ''"
    )
    # Now that the columns exist, create indexes that reference them.
    # These can't live in SCHEMA_SQL because executescript runs them
    # BEFORE this migration adds the columns on pre-v1.9 DBs.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS code_finding_fp_idx     ON code_finding(fingerprint)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS code_finding_triage_idx ON code_finding(triage_state)"
    )


def compute_finding_fingerprint(
    *, tool: str, file_path: str, kind: str,
    rule_id: Optional[str], excerpt: str,
) -> str:
    """Stable hash so the same finding across multiple scans is recognised
    as the SAME finding (used for drift detection).

    Inputs are normalised:
      - `file_path` is forward-slashed
      - `excerpt` is first 80 chars, whitespace-collapsed, lowercased
      - `tool` strips the 'external:' prefix
      - `rule_id` is the rule label if available, else falls back to `kind`
    """
    import hashlib as _h
    norm_path = (file_path or "").replace("\\", "/").strip().lower()
    norm_excerpt = " ".join((excerpt or "").split())[:80].lower()
    norm_tool = (tool or "").removeprefix("external:").strip().lower()
    norm_rule = (rule_id or kind or "").strip().lower()
    h = _h.sha256()
    h.update(f"{norm_tool}|{norm_path}|{kind.lower()}|{norm_rule}|{norm_excerpt}".encode("utf-8"))
    return h.hexdigest()[:24]


VALID_TRIAGE_STATES = (
    "new", "confirmed", "false_positive", "fixed", "wontfix", "suppressed",
)


def triage_finding(
    conn, finding_id: int, *, state: str,
    note: Optional[str] = None, by: Optional[str] = None,
) -> Dict[str, Any]:
    """Set or update the triage state of a single finding.

    Setting state='fixed' also stamps `fixed_at = now` (idempotently --
    if already fixed, fixed_at is preserved).
    """
    s = (state or "").strip().lower()
    if s not in VALID_TRIAGE_STATES:
        raise ValueError(f"invalid triage state: {state!r}")
    now = _utcnow()
    cur = conn.execute(
        "SELECT triage_state, fixed_at, created_at FROM code_finding WHERE id = ?",
        (finding_id,),
    ).fetchone()
    if not cur:
        raise KeyError(f"finding {finding_id} not found")
    fixed_at = cur["fixed_at"]
    if s == "fixed" and not fixed_at:
        fixed_at = now
    elif s != "fixed":
        # If we move out of 'fixed' (e.g. reopened), clear fixed_at.
        if cur["triage_state"] == "fixed":
            fixed_at = None
    conn.execute(
        "UPDATE code_finding SET triage_state=?, triage_note=?, triage_by=?, "
        "triage_at=?, fixed_at=? WHERE id=?",
        (s, note, by, now, fixed_at, finding_id),
    )
    return {"finding_id": finding_id, "triage_state": s,
            "triage_at": now, "fixed_at": fixed_at}


def scanner_obs_summary(conn, *, since: Optional[str] = None) -> Dict[str, Any]:
    """Top-level KPIs across the whole scanner fleet.

    Args:
      since: ISO timestamp lower bound on `code_scan.started_at`.
    Returns: dict of KPIs (see Scanner Obs UI tab).
    """
    where = ["status = 'done'"]
    args: List[Any] = []
    if since:
        where.append("started_at >= ?")
        args.append(since)
    where_sql = " AND ".join(where)
    # Per-scan counters
    scan_stats = conn.execute(
        f"SELECT COUNT(*) AS n_scans, "
        f"  SUM(findings_count) AS total_findings, "
        f"  SUM(cost_usd)        AS total_cost_usd, "
        f"  SUM(ci_minutes)      AS total_ci_minutes, "
        f"  SUM(elapsed_ms)      AS total_elapsed_ms "
        f"FROM code_scan WHERE {where_sql}",
        args,
    ).fetchone()
    n_scans         = scan_stats["n_scans"] or 0
    total_findings  = scan_stats["total_findings"] or 0
    total_cost      = scan_stats["total_cost_usd"]
    # $/finding only when BOTH numerator and denominator are non-null
    dollar_per_finding: Optional[float] = None
    if total_cost is not None and total_findings > 0:
        dollar_per_finding = round(float(total_cost) / float(total_findings), 6)

    # Triage stats -- ONLY across findings whose scan matches the window
    f_where = ["1=1"]
    f_args: List[Any] = []
    if since:
        f_where.append("cs.started_at >= ?")
        f_args.append(since)
    f_where_sql = " AND ".join(f_where)
    triage_rows = conn.execute(
        f"SELECT cf.triage_state AS s, COUNT(*) AS n "
        f"FROM code_finding cf JOIN code_scan cs ON cs.id = cf.scan_id "
        f"WHERE {f_where_sql} GROUP BY cf.triage_state",
        f_args,
    ).fetchall()
    triage = {r["s"] or "new": r["n"] for r in triage_rows}
    triaged_real = triage.get("confirmed", 0) + triage.get("false_positive", 0) \
                 + triage.get("fixed", 0) + triage.get("wontfix", 0)
    fp = triage.get("false_positive", 0)
    fp_rate: Optional[float] = None
    # Only report FP rate once we have a meaningful triage sample. Below
    # 5 triaged findings the number is too noisy to surface honestly.
    if triaged_real >= 5:
        fp_rate = round(fp / triaged_real, 4)

    # Time-to-fix percentiles (julianday delta in days)
    ttf_rows = conn.execute(
        f"SELECT (julianday(cf.fixed_at) - julianday(cf.created_at)) * 86400.0 AS sec "
        f"FROM code_finding cf JOIN code_scan cs ON cs.id = cf.scan_id "
        f"WHERE cf.fixed_at IS NOT NULL AND {f_where_sql} "
        f"ORDER BY sec",
        f_args,
    ).fetchall()
    ttf_p50: Optional[float] = None
    ttf_p90: Optional[float] = None
    if len(ttf_rows) >= 3:
        secs = [float(r["sec"]) for r in ttf_rows if r["sec"] is not None]
        if secs:
            secs.sort()
            ttf_p50 = secs[int(0.50 * (len(secs) - 1))]
            ttf_p90 = secs[int(0.90 * (len(secs) - 1))]

    return {
        "since":             since,
        "n_scans":           n_scans,
        "total_findings":    total_findings,
        "total_cost_usd":    total_cost,
        "total_ci_minutes":  scan_stats["total_ci_minutes"],
        "dollar_per_finding": dollar_per_finding,
        "triage":            triage,
        "fp_rate":           fp_rate,
        "ttf_p50_seconds":   ttf_p50,
        "ttf_p90_seconds":   ttf_p90,
    }


def scanner_obs_per_tool(conn, *, since: Optional[str] = None) -> List[Dict[str, Any]]:
    """Per-tool breakdown: one row per distinct `code_scan.model` value
    (e.g. 'external:semgrep', 'external:codeql', 'qwen2.5-coder:3b').
    """
    where = ["cs.status = 'done'"]
    args: List[Any] = []
    if since:
        where.append("cs.started_at >= ?")
        args.append(since)
    where_sql = " AND ".join(where)
    # scanner_version is intentionally picked from the *most recent*
    # scan of each tool, not MAX(version): MAX on a TEXT column does
    # lexical comparison, which gives wrong answers as soon as version
    # strings vary in format (e.g. "1.50.0" vs "semgrep 1.45.0" -- the
    # latter sorts higher because 's' > '1').
    rows = conn.execute(
        f"SELECT cs.model AS tool, "
        f"  COUNT(DISTINCT cs.id)  AS n_scans, "
        f"  SUM(cs.findings_count) AS n_findings, "
        f"  SUM(cs.cost_usd)       AS cost_usd, "
        f"  (SELECT scanner_version FROM code_scan cs2 "
        f"     WHERE cs2.model = cs.model AND cs2.status = 'done' "
        f"     ORDER BY cs2.started_at DESC LIMIT 1) AS scanner_version, "
        f"  MAX(cs.finished_at)    AS last_run "
        f"FROM code_scan cs WHERE {where_sql} "
        f"GROUP BY cs.model ORDER BY n_findings DESC",
        args,
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        tool = r["tool"]
        # Triage stats per tool
        tri = conn.execute(
            "SELECT cf.triage_state AS s, COUNT(*) AS n "
            "FROM code_finding cf JOIN code_scan cs ON cs.id = cf.scan_id "
            "WHERE cs.model = ? GROUP BY cf.triage_state",
            (tool,),
        ).fetchall()
        triage = {t["s"] or "new": t["n"] for t in tri}
        triaged = (triage.get("confirmed", 0) + triage.get("false_positive", 0)
                   + triage.get("fixed", 0) + triage.get("wontfix", 0))
        fp = triage.get("false_positive", 0)
        fp_rate = round(fp / triaged, 4) if triaged >= 5 else None
        n_findings = r["n_findings"] or 0
        dpf = (round(float(r["cost_usd"]) / n_findings, 6)
               if r["cost_usd"] is not None and n_findings else None)
        out.append({
            "tool":               tool,
            "n_scans":            r["n_scans"],
            "n_findings":         n_findings,
            "cost_usd":           r["cost_usd"],
            "dollar_per_finding": dpf,
            "scanner_version":    r["scanner_version"],
            "last_run":           r["last_run"],
            "triage":             triage,
            "fp_rate":            fp_rate,
        })
    return out


def scanner_obs_drift(
    conn, *, tool: str, root_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare the two most-recent scans of (tool, root_path).

    Returns:
        {
          "previous": {scan_id, started_at} or None,
          "latest":   {scan_id, started_at} or None,
          "new":         [fingerprints in latest only],
          "persistent":  [fingerprints in both],
          "gone":        [fingerprints in previous only],
        }
    """
    where = ["status = 'done'", "model = ?"]
    args: List[Any] = [tool]
    if root_path:
        where.append("root_path = ?")
        args.append(root_path)
    where_sql = " AND ".join(where)
    scans = conn.execute(
        f"SELECT id, started_at, root_path FROM code_scan WHERE {where_sql} "
        f"ORDER BY started_at DESC LIMIT 2", args,
    ).fetchall()
    if len(scans) < 2:
        return {
            "previous": None,
            "latest":   (dict(scans[0]) if scans else None),
            "new": [], "persistent": [], "gone": [],
            "note": "need at least 2 completed scans for drift",
        }
    latest, prev = scans[0], scans[1]
    def _fps(scan_id):
        rows = conn.execute(
            "SELECT id, fingerprint, file_path, kind, severity, triage_state "
            "FROM code_finding WHERE scan_id = ? AND fingerprint IS NOT NULL",
            (scan_id,),
        ).fetchall()
        return {r["fingerprint"]: dict(r) for r in rows}
    a = _fps(latest["id"])
    b = _fps(prev["id"])
    new        = [a[k] for k in a.keys() - b.keys()]
    gone       = [b[k] for k in b.keys() - a.keys()]
    persistent = [a[k] for k in a.keys() & b.keys()]
    return {
        "previous":   dict(prev),
        "latest":     dict(latest),
        "new":        new,
        "persistent": persistent,
        "gone":       gone,
    }


def scanner_obs_density(
    conn, *, since: Optional[str] = None, top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Findings density per `kind`. Returns the top-N kinds by count,
    each with a triage breakdown so the user can see which kinds the
    fleet currently treats as noise."""
    where = ["1=1"]
    args: List[Any] = []
    if since:
        where.append("cs.started_at >= ?")
        args.append(since)
    where_sql = " AND ".join(where)
    rows = conn.execute(
        f"SELECT cf.kind AS kind, COUNT(*) AS n, "
        f"  SUM(CASE WHEN cf.triage_state='false_positive' THEN 1 ELSE 0 END) AS fp, "
        f"  SUM(CASE WHEN cf.triage_state='confirmed'      THEN 1 ELSE 0 END) AS confirmed, "
        f"  SUM(CASE WHEN cf.triage_state='fixed'          THEN 1 ELSE 0 END) AS fixed_c, "
        f"  SUM(CASE WHEN cf.triage_state IN ('new', '') OR cf.triage_state IS NULL THEN 1 ELSE 0 END) AS new_c, "
        f"  AVG(cf.severity_rank) AS avg_rank "
        f"FROM code_finding cf JOIN code_scan cs ON cs.id = cf.scan_id "
        f"WHERE {where_sql} GROUP BY cf.kind "
        f"ORDER BY n DESC LIMIT ?",
        args + [top_n],
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# CRUD helpers (small, intentional surface)
# ---------------------------------------------------------------------------

def upsert_agent(
    conn, name: str, description: str = "", *, kind: str = "qwen-vllm",
) -> int:
    """Get-or-create an agent row.

    `kind` identifies which runtime produces traces for this agent:
    qwen-vllm | openai | anthropic | ollama | langchain | autogen |
    smolagents | custom. The dashboard uses it to decide whether the
    Interp tab is meaningful (only Qwen-family models have probes).

    If the agent already exists, its `kind` is updated (so renaming a
    runtime doesn't require manually fixing the DB).
    """
    cur = conn.execute("SELECT id, kind FROM agent WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        if row["kind"] != kind:
            conn.execute("UPDATE agent SET kind = ? WHERE id = ?",
                         (kind, int(row["id"])))
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO agent (name, description, kind, created_at) "
        "VALUES (?, ?, ?, ?)",
        (name, description, kind, _utcnow()),
    )
    return int(cur.lastrowid)


def create_run(
    conn, agent_id: int, *, external_id: Optional[str] = None,
    input_text: str = "", meta: Optional[Dict[str, Any]] = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO run (agent_id, external_id, status, input_text, meta_json, started_at) "
        "VALUES (?, ?, 'running', ?, ?, ?)",
        (agent_id, external_id, input_text,
         json.dumps(meta or {}), _utcnow()),
    )
    return int(cur.lastrowid)


def finish_run(
    conn, run_id: int, *, status: str, output_text: str = "",
    elapsed_ms: Optional[int] = None, meta: Optional[Dict[str, Any]] = None,
    tokens_in: Optional[int] = None, tokens_out: Optional[int] = None,
    model_id: Optional[str] = None, cost_usd: Optional[float] = None,
) -> None:
    """Mark a run terminal. v1.7+: optionally persist economics columns.

    All new args are None-tolerant. Passing None for any of them means
    "don't update this column" -- existing values are preserved via
    COALESCE so adapters can finish a run incrementally without
    clobbering token counts recorded mid-run.
    """
    if meta is not None:
        # merge with existing meta_json
        cur = conn.execute("SELECT meta_json FROM run WHERE id = ?", (run_id,))
        row = cur.fetchone()
        prev = json.loads(row["meta_json"] or "{}") if row else {}
        prev.update(meta)
        meta_json = json.dumps(prev)
    else:
        meta_json = None
    conn.execute(
        "UPDATE run SET status = ?, output_text = ?, elapsed_ms = ?, "
        "finished_at = ?, "
        "meta_json  = COALESCE(?, meta_json), "
        "tokens_in  = COALESCE(?, tokens_in), "
        "tokens_out = COALESCE(?, tokens_out), "
        "model_id   = COALESCE(?, model_id), "
        "cost_usd   = COALESCE(?, cost_usd) "
        "WHERE id = ?",
        (status, output_text, elapsed_ms, _utcnow(), meta_json,
         tokens_in, tokens_out, model_id, cost_usd, run_id),
    )


def append_trace(
    conn, run_id: int, kind: str, payload: Dict[str, Any],
) -> int:
    cur = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM trace_event WHERE run_id = ?",
        (run_id,),
    )
    seq = int(cur.fetchone()[0])
    cur = conn.execute(
        "INSERT INTO trace_event (run_id, seq, kind, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, seq, kind, json.dumps(payload, default=str), _utcnow()),
    )
    return int(cur.lastrowid)


def record_interp_score(
    conn, run_id: int, *, target: str, probe: str, score: float,
) -> None:
    conn.execute(
        "INSERT INTO interp_score (run_id, target, probe, score, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, target, probe, float(score), _utcnow()),
    )


def add_memory(
    conn, *, source: str, kind: str, text: str,
    embedding: Optional[bytes] = None, embed_dim: Optional[int] = None,
    tags: Iterable[str] = (),
) -> int:
    cur = conn.execute(
        "INSERT INTO memory_chunk (source, kind, text, embedding_blob, embed_dim, tags, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, kind, text, embedding, embed_dim, ",".join(tags), _utcnow()),
    )
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Read helpers (used by API + smoke)
# ---------------------------------------------------------------------------

def list_agents(conn) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM agent ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def list_runs(conn, *, limit: int = 50, agent_id: Optional[int] = None) -> List[Dict[str, Any]]:
    if agent_id is None:
        rows = conn.execute(
            "SELECT * FROM run ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM run WHERE agent_id = ? ORDER BY started_at DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def list_trace(conn, run_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM trace_event WHERE run_id = ? ORDER BY seq", (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def list_interp_scores(conn, run_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM interp_score WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def record_nla_decoding(
    conn, *, run_id: Optional[int], target: str, decoding: Dict[str, Any],
    trace_seq: Optional[int] = None,
) -> int:
    """Persist a single NLA decoding from agent_monitor.nla_client.decode().

    `decoding` is the dict returned by nla_client.decode(); we store the
    canonical fields plus the full payload as raw_json for audit.
    """
    notes = decoding.get("notes")
    notes_json = json.dumps(notes) if notes is not None else None
    raw_json = json.dumps(
        {k: v for k, v in decoding.items() if k != "raw"},
        default=str,
    )
    cur = conn.execute(
        "INSERT INTO nla_decoding (run_id, trace_seq, target, backend, model, "
        "explanation, topic, evaluation_awareness, hidden_motivation, "
        "safety_relevance, notes_json, raw_json, latency_ms, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id, trace_seq, target,
            decoding.get("source") or "unknown",
            decoding.get("model") or decoding.get("target_model"),
            decoding.get("explanation"),
            decoding.get("topic"),
            decoding.get("evaluation_awareness"),
            decoding.get("hidden_motivation"),
            decoding.get("safety_relevance"),
            notes_json,
            raw_json,
            decoding.get("latency_ms"),
            _utcnow(),
        ),
    )
    return int(cur.lastrowid)


def list_nla_decodings(conn, run_id: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM nla_decoding WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# v1.5: code scan helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK_DB = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def create_code_scan(
    conn, *, root_path: str, label: Optional[str], model: Optional[str],
    prompt_version: Optional[str], options: Dict[str, Any],
) -> int:
    cur = conn.execute(
        "INSERT INTO code_scan (root_path, label, status, model, "
        "prompt_version, options_json, started_at) "
        "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
        (
            root_path, label, model, prompt_version,
            json.dumps(options, default=str),
            _utcnow(),
        ),
    )
    return int(cur.lastrowid)


def update_code_scan(
    conn, scan_id: int, *,
    status: Optional[str] = None,
    total_files: Optional[int] = None,
    scanned_files: Optional[int] = None,
    skipped_files: Optional[int] = None,
    findings_count: Optional[int] = None,
    error: Optional[str] = None,
    cost_usd: Optional[float] = None,
    ci_minutes: Optional[float] = None,
    scanner_version: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
    finished: bool = False,
) -> None:
    sets: List[str] = []
    vals: List[Any] = []
    for col, val in (
        ("status", status), ("total_files", total_files),
        ("scanned_files", scanned_files), ("skipped_files", skipped_files),
        ("findings_count", findings_count), ("error", error),
        ("cost_usd", cost_usd), ("ci_minutes", ci_minutes),
        ("scanner_version", scanner_version), ("elapsed_ms", elapsed_ms),
    ):
        if val is not None:
            sets.append(f"{col} = ?")
            vals.append(val)
    if finished:
        sets.append("finished_at = ?")
        vals.append(_utcnow())
    if not sets:
        return
    vals.append(scan_id)
    conn.execute(
        f"UPDATE code_scan SET {', '.join(sets)} WHERE id = ?", vals,
    )


def record_code_finding(
    conn, *, scan_id: int, file_path: str, file_sha256: Optional[str],
    chunk_index: int, chunk_start_line: int, chunk_end_line: int,
    language: Optional[str], kind: str, severity: str,
    line_hint: Optional[int], excerpt: str, explanation: Optional[str],
    chunk_summary: Optional[str],
    fingerprint: Optional[str] = None,
) -> int:
    sev = (severity or "info").strip().lower()
    rank = _SEVERITY_RANK_DB.get(sev, 0)
    cur = conn.execute(
        "INSERT INTO code_finding (scan_id, file_path, file_sha256, "
        "chunk_index, chunk_start_line, chunk_end_line, language, kind, "
        "severity, severity_rank, line_hint, excerpt, explanation, "
        "chunk_summary, created_at, fingerprint, triage_state) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')",
        (
            scan_id, file_path, file_sha256, chunk_index,
            chunk_start_line, chunk_end_line, language, kind, sev, rank,
            line_hint, excerpt, explanation, chunk_summary, _utcnow(),
            fingerprint,
        ),
    )
    return int(cur.lastrowid)


def get_code_scan(conn, scan_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM code_scan WHERE id = ?", (scan_id,)
    ).fetchone()
    return dict(row) if row else None


def list_code_scans(conn, *, limit: int = 50) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM code_scan ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def list_sandbox_detonations(conn, *, limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most-recent sandbox detonations, one row per ``code_scan``.

    A *detonation* is the v1.10 ingest output: an external scan whose
    findings all share one ``file_path`` (the sample SHA-256 or filename).
    We discriminate sandbox scans from regular static-analysis scans by
    requiring at least one finding whose ``file_path`` starts with
    ``sha256:`` -- the canonical marker that
    ``agent_monitor.adapters.sandbox_report._sample_id`` produces when a
    SHA-256 is present in the report.

    Each returned row carries:
      * scan_id, model (= ``external:<tool>``), scanner_version
      * label, started_at, status, root_path
      * sample_id           -- the SHA-256-prefixed file_path (or filename
                               fallback)
      * n_findings          -- total signatures persisted from that scan
      * severity            -- {info/low/medium/high/critical: count}

    The query is two passes: list scans, then compute the per-scan
    histogram. Reasonable for typical scale (<10k detonations); if that
    ever stops being true we'd switch to a GROUP BY subquery.
    """
    scan_rows = conn.execute(
        "SELECT cs.* FROM code_scan cs "
        "WHERE cs.model LIKE 'external:%' "
        "  AND EXISTS ("
        "      SELECT 1 FROM code_finding cf "
        "      WHERE cf.scan_id = cs.id "
        "        AND cf.file_path LIKE 'sha256:%'"
        "  ) "
        "ORDER BY cs.id DESC LIMIT ?",
        (limit,),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in scan_rows:
        d = dict(r)
        scan_id = d["id"]
        # Pick one sample identifier (all findings share file_path on a
        # sandbox scan). MIN gives us a deterministic pick.
        sample_row = conn.execute(
            "SELECT MIN(file_path) AS sample_id FROM code_finding "
            "WHERE scan_id = ?", (scan_id,),
        ).fetchone()
        sample_id = sample_row["sample_id"] if sample_row else None
        # Severity histogram (5 buckets, all keys always present).
        hist = {"info": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
        sev_rows = conn.execute(
            "SELECT severity, COUNT(*) AS n FROM code_finding "
            "WHERE scan_id = ? GROUP BY severity",
            (scan_id,),
        ).fetchall()
        for sr in sev_rows:
            if sr["severity"] in hist:
                hist[sr["severity"]] = sr["n"]
        d["sample_id"] = sample_id
        d["severity"]  = hist
        # Strip ``external:`` so the UI can render a short tool label.
        d["tool"] = (d.get("model") or "").removeprefix("external:") or "?"
        out.append(d)
    return out


def list_code_findings(
    conn, scan_id: int, *,
    min_severity: Optional[str] = None,
    kind: Optional[str] = None,
    file_path: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    sql = ["SELECT * FROM code_finding WHERE scan_id = ?"]
    params: List[Any] = [scan_id]
    if min_severity:
        rank = _SEVERITY_RANK_DB.get(min_severity.lower(), 0)
        sql.append("AND severity_rank >= ?")
        params.append(rank)
    if kind:
        sql.append("AND kind = ?")
        params.append(kind)
    if file_path:
        sql.append("AND file_path = ?")
        params.append(file_path)
    sql.append("ORDER BY severity_rank DESC, id DESC LIMIT ?")
    params.append(limit)
    rows = conn.execute(" ".join(sql), params).fetchall()
    return [dict(r) for r in rows]


def code_scan_severity_histogram(conn, scan_id: int) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT severity, COUNT(*) AS n FROM code_finding "
        "WHERE scan_id = ? GROUP BY severity",
        (scan_id,),
    ).fetchall()
    hist = {"info": 0, "low": 0, "medium": 0, "high": 0, "critical": 0}
    for r in rows:
        hist[r["severity"]] = r["n"]
    return hist
