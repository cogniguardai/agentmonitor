"""
External findings ingest -- pipe ANY static-analysis tool's output
(Semgrep, CodeQL, Bandit, ESLint, your homegrown linter, an LLM-based
checker) into AgentMonitor's `code_finding` table.

WHY THIS EXISTS
===============

AgentMonitor already has a `code_scan` / `code_finding` schema and a
Code Scan tab that renders both. The built-in scanner uses an LLM (via
`agent_monitor.code_scan`), but the schema is generic. Letting external
tools write into the same surface gives users a single pane of glass:

    "I ran Semgrep at 09:00, my LLM scanner at 09:30, and CodeQL
     overnight -- show me ALL findings sorted by severity."

This module owns NO detection logic. We don't ship rules, signatures,
or heuristics. The user's tool already detected the issues; we just
persist them in a uniform shape.

WHAT WE DO NOT DO
=================

* We do NOT write Semgrep / CodeQL / Bandit rules.
* We do NOT classify whether a finding is exploitable, weaponizable,
  or otherwise interesting beyond what the tool itself says. The
  `severity` field is whatever the tool reported -- we don't relabel.
* We do NOT enrich findings with exploit hints, primitive analysis,
  or anything that turns "static-analysis warning" into "exploit
  candidate." Those lines live in `agent_monitor`'s scope doc.

USAGE (Python helper)
=====================

::

    from agent_monitor.adapters.findings import ExternalScan

    with ExternalScan(tool_name="semgrep",
                      root_path=r"C:\\src\\my-app",
                      label="weekly semgrep") as scan:
        scan.add(file_path="src/auth.py",
                 kind="auth-bypass",
                 severity="high",
                 line=42,
                 excerpt="if user == 'admin': return True",
                 message="hard-coded admin bypass")
        # ...

The scan automatically appears in the Code Scan tab.

USAGE (REST, for non-Python tools)
==================================

POST /api/code_scan/external
{
  "tool_name": "semgrep",
  "root_path": "C:\\src\\my-app",
  "label":     "weekly semgrep",
  "findings":  [
      {"file_path": "src/auth.py", "kind": "auth-bypass",
       "severity": "high", "line": 42,
       "excerpt": "if user == 'admin': return True",
       "message": "hard-coded admin bypass"},
      ...
  ]
}

SEMGREP CONVENIENCE
===================

`parse_semgrep_output(json_str)` maps Semgrep's standard `--json` output
shape to AgentMonitor's finding fields. This is plumbing, not detection
-- the rules / signatures live in the user's Semgrep config.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from agent_monitor import db

_VALID_SEVERITY = ("info", "low", "medium", "high", "critical")


def _normalize_severity(s: Optional[str]) -> str:
    """Map common provider severity names to our enum. We keep the
    mapping conservative -- when in doubt we stay LOWER, not higher.
    A bug-aware user can pass severity verbatim by using one of our
    enum values."""
    if not s:
        return "info"
    s = str(s).strip().lower()
    if s in _VALID_SEVERITY:
        return s
    # Common Semgrep / CodeQL / SARIF variants
    return {
        "warning":      "medium",
        "error":        "high",
        "note":         "info",
        "recommendation":"low",
        "minor":        "low",
        "major":        "medium",
        "blocker":      "critical",
    }.get(s, "info")


# ---------------------------------------------------------------------------
# Python helper (the recommended surface)
# ---------------------------------------------------------------------------

@contextmanager
def ExternalScan(
    *, tool_name: str, root_path: str,
    label: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    # v1.9: scanner-observability fields. All optional, all NULL when
    # unknown -- we never invent a number.
    cost_usd:        Optional[float] = None,
    ci_minutes:      Optional[float] = None,
    scanner_version: Optional[str]   = None,
    elapsed_ms:      Optional[int]   = None,
) -> Iterator["ExternalScanHandle"]:
    """Open a `code_scan` row tagged `external:<tool_name>` and hand
    back a handle for adding findings. Closes the row to status='done'
    on context exit (or 'error' on exception).

    The cost/version fields populate the Scanner Obs dashboard. They are
    OPTIONAL -- if the caller doesn't know the cost they pass None and
    the UI shows '—' rather than zero.
    """
    opts = dict(options or {})
    opts["external_tool"] = tool_name
    with db.session() as conn:
        scan_id = db.create_code_scan(
            conn,
            root_path=root_path,
            label=label or f"{tool_name} scan",
            model=f"external:{tool_name}",
            prompt_version=None,
            options=opts,
        )
        # Mark running + stamp cost/version up front so even crashed
        # scans report what they were configured with.
        db.update_code_scan(
            conn, scan_id, status="running",
            cost_usd=cost_usd, ci_minutes=ci_minutes,
            scanner_version=scanner_version, elapsed_ms=elapsed_ms,
        )
    handle = ExternalScanHandle(scan_id, tool_name=tool_name, root_path=root_path)
    try:
        yield handle
    except Exception as e:
        with db.session() as conn:
            db.update_code_scan(
                conn, scan_id, status="error",
                error=f"{type(e).__name__}: {e}",
                finished=True,
            )
        raise
    else:
        with db.session() as conn:
            db.update_code_scan(
                conn, scan_id, status="done",
                scanned_files=handle.unique_files,
                findings_count=handle.n_findings,
                finished=True,
            )


class ExternalScanHandle:
    def __init__(self, scan_id: int, *, tool_name: str, root_path: str):
        self.scan_id = scan_id
        self.tool_name = tool_name
        self.root_path = root_path
        self.n_findings = 0
        self._files: set = set()

    @property
    def unique_files(self) -> int:
        return len(self._files)

    def add(
        self, *, file_path: str, kind: str,
        severity: str = "info",
        line: Optional[int] = None,
        end_line: Optional[int] = None,
        excerpt: str = "",
        message: str = "",
        rule_id: Optional[str] = None,
        language: Optional[str] = None,
        file_sha256: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist one finding. Returns the new `code_finding.id`.

        `kind` is the tool's category label (e.g. "auth-bypass",
        "buffer-overflow", "missing-encoding"). We do NOT validate it
        against a known taxonomy because every tool has its own.

        `excerpt` should be a short verbatim quote from the file (a few
        lines max). Long excerpts are truncated to 4000 chars to keep
        the DB sane.
        """
        sev = _normalize_severity(severity)
        excerpt = (excerpt or message)[:4000]
        explanation = message[:2000] if message else None
        chunk_summary = None
        if rule_id or extra:
            chunk_summary = json.dumps({
                "rule_id": rule_id, "tool": self.tool_name, "extra": extra or {},
            })[:2000]
        line_hint = int(line) if line is not None else None
        start_line = int(line) if line is not None else 1
        end = int(end_line) if end_line is not None else (start_line or 1)
        if end < start_line:
            end = start_line
        # v1.9: stable cross-scan fingerprint for drift detection.
        fingerprint = db.compute_finding_fingerprint(
            tool=self.tool_name, file_path=file_path, kind=kind,
            rule_id=rule_id, excerpt=excerpt,
        )
        with db.session() as conn:
            fid = db.record_code_finding(
                conn,
                scan_id=self.scan_id,
                file_path=file_path,
                file_sha256=file_sha256,
                chunk_index=0,
                chunk_start_line=start_line,
                chunk_end_line=end,
                language=language,
                kind=kind,
                severity=sev,
                line_hint=line_hint,
                excerpt=excerpt,
                explanation=explanation,
                chunk_summary=chunk_summary,
                fingerprint=fingerprint,
            )
        self.n_findings += 1
        self._files.add(file_path)
        return fid


# ---------------------------------------------------------------------------
# Bulk helper -- write a whole list at once
# ---------------------------------------------------------------------------

def ingest_findings(
    *, tool_name: str, root_path: str, findings: List[Dict[str, Any]],
    label: Optional[str] = None,
    cost_usd:        Optional[float] = None,
    ci_minutes:      Optional[float] = None,
    scanner_version: Optional[str]   = None,
    elapsed_ms:      Optional[int]   = None,
) -> Dict[str, Any]:
    """Single-call ingest of a list of findings. Used by the REST
    endpoint and convenient from one-shot scripts.

    Optional v1.9 args (cost_usd, ci_minutes, scanner_version,
    elapsed_ms) populate the Scanner Obs dashboard.
    """
    with ExternalScan(
        tool_name=tool_name, root_path=root_path, label=label,
        cost_usd=cost_usd, ci_minutes=ci_minutes,
        scanner_version=scanner_version, elapsed_ms=elapsed_ms,
    ) as scan:
        for f in findings:
            scan.add(
                file_path=str(f.get("file_path") or f.get("path") or "?"),
                kind=str(f.get("kind") or f.get("rule") or f.get("check_id") or "external"),
                severity=str(f.get("severity") or "info"),
                line=f.get("line") or f.get("start_line"),
                end_line=f.get("end_line"),
                excerpt=str(f.get("excerpt") or f.get("snippet") or ""),
                message=str(f.get("message") or ""),
                rule_id=f.get("rule_id") or f.get("check_id"),
                language=f.get("language"),
                file_sha256=f.get("file_sha256"),
                extra=f.get("extra"),
            )
        return {
            "scan_id":        scan.scan_id,
            "n_findings":     scan.n_findings,
            "unique_files":   scan.unique_files,
        }


# ---------------------------------------------------------------------------
# SARIF convenience -- the universal path. Any SARIF-emitting tool
# (Semgrep, CodeQL, Bandit, Snyk, Trivy, ESLint, Checkov, Gitleaks, ...)
# flows through here with no per-tool code.
# ---------------------------------------------------------------------------

def ingest_sarif(
    payload: Any, *,
    root_path: str,
    label:           Optional[str] = None,
    tool_override:   Optional[str] = None,
    cost_usd:        Optional[float] = None,
    ci_minutes:      Optional[float] = None,
    elapsed_ms:      Optional[int]   = None,
) -> Dict[str, Any]:
    """Ingest a SARIF v2.1.0 document.

    One SARIF document can contain multiple `runs` (typically each is a
    separate tool invocation). Each run becomes its own `code_scan`
    row, so the Code Scan tab and Scanner Obs KPIs stay per-tool.

    Args:
        payload:       SARIF as a dict or JSON string.
        root_path:     The repo / project root the scan applied to.
                       Used for drift detection (drift compares scans
                       sharing the same tool + root_path).
        label:         Optional human label. Defaults to
                       `sarif:<tool> #<run-index>`.
        tool_override: Force a specific tool name. Useful when the
                       driver name is generic (e.g. some wrappers emit
                       `tool.driver.name = "sarif"`).
        cost_usd, ci_minutes, elapsed_ms:
                       Scanner Obs fields. If multiple runs are in the
                       same document, these are split evenly across
                       runs (assumption: caller measured ONE tool
                       invocation that produced this whole file).

    Returns:
        {
          "n_runs":  <int>,
          "runs":    [ <one ingest_findings result per SARIF run>, ... ],
        }

    Raises:
        ValueError: if the payload isn't valid SARIF.
    """
    from agent_monitor.adapters.sarif import parse_sarif
    runs = parse_sarif(payload)
    n_runs = max(len(runs), 1)

    def _share(v):
        return None if v is None else (v / n_runs)

    results: List[Dict[str, Any]] = []
    for i, run in enumerate(runs):
        out = ingest_findings(
            tool_name=tool_override or run["tool"],
            root_path=root_path,
            label=label or f"sarif:{run['tool']} #{i + 1}",
            findings=run["findings"],
            scanner_version=run["scanner_version"],
            cost_usd=_share(cost_usd),
            ci_minutes=_share(ci_minutes),
            elapsed_ms=(None if elapsed_ms is None else int(elapsed_ms / n_runs)),
        )
        out["tool"] = run["tool"]
        out["scanner_version"] = run["scanner_version"]
        out["n_results"] = run["n_results"]
        results.append(out)
    return {"n_runs": len(results), "runs": results}


# ---------------------------------------------------------------------------
# Sandbox-report convenience -- the *dynamic*-analysis sister to ingest_sarif.
# Any sandbox (Cuckoo / Joe / VMRay / ANY.RUN / homegrown) flows through here.
# ---------------------------------------------------------------------------

def ingest_sandbox_report(
    payload: Any, *,
    root_path: str,
    label:           Optional[str] = None,
    format:          Optional[str] = None,
    tool_override:   Optional[str] = None,
    cost_usd:        Optional[float] = None,
    ci_minutes:      Optional[float] = None,
    elapsed_ms:      Optional[int]   = None,
) -> Dict[str, Any]:
    """Ingest a VM/sandbox JSON report (Cuckoo or generic envelope).

    Each report describes ONE detonated sample, so unlike SARIF this
    always produces exactly one `code_scan` row. The sample's SHA-256
    (or filename, if no SHA is present) is what `code_finding.file_path`
    holds; the sandbox's signatures / signals each become one finding.

    Args:
        payload:       Sandbox report as a dict or JSON string.
        root_path:     Identifier for the *batch* of sandbox runs --
                       drift detection groups by ``tool + root_path``,
                       so use a stable label here (e.g. ``"sandbox/prod"``
                       or the analysis queue name) rather than a per-run
                       directory.
        label:         Human-readable scan label. Defaults to
                       ``sandbox:<tool> <sample_id>``.
        format:        Optional explicit format -- ``"cuckoo"`` or
                       ``"generic"``. Auto-detected when omitted.
        tool_override: Force a specific tool name. Useful when several
                       sandbox versions share one report shape and the
                       caller wants them tracked separately.
        cost_usd, ci_minutes, elapsed_ms:
                       Scanner-Obs cost / runtime fields. We pass them
                       straight through; sandbox detonations are
                       single-run by definition, so no splitting is
                       needed (unlike `ingest_sarif`).

    Returns:
        ``ingest_findings`` result dict, plus
        ``{"tool", "scanner_version", "sample_id", "n_signals"}`` for
        callers / tests that want the parsed envelope summary.

    Raises:
        ValueError: if the payload isn't a recognized sandbox report.
    """
    from agent_monitor.adapters.sandbox_report import parse_sandbox_report
    parsed = parse_sandbox_report(payload, format=format)
    tool_name = tool_override or parsed["tool"]
    sample_id = parsed["sample_id"]
    out = ingest_findings(
        tool_name=tool_name,
        root_path=root_path,
        label=label or f"sandbox:{parsed['tool']} {sample_id}",
        findings=parsed["findings"],
        scanner_version=parsed["scanner_version"],
        cost_usd=cost_usd,
        ci_minutes=ci_minutes,
        elapsed_ms=elapsed_ms,
    )
    out["tool"] = tool_name
    out["scanner_version"] = parsed["scanner_version"]
    out["sample_id"] = sample_id
    out["n_signals"] = parsed["n_signals"]
    return out


# ---------------------------------------------------------------------------
# Semgrep convenience -- map their JSON to ours
# ---------------------------------------------------------------------------

def parse_semgrep_output(payload: Any) -> List[Dict[str, Any]]:
    """Convert `semgrep --json` output (dict with a `results` key) into
    a list of finding dicts suitable for `ingest_findings`.

    This is plumbing only -- the rules / detection logic live in the
    user's Semgrep config. We translate field names, nothing else.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    out: List[Dict[str, Any]] = []
    for r in results:
        extra = r.get("extra") or {}
        out.append({
            "file_path":  r.get("path") or "?",
            "kind":       r.get("check_id") or "semgrep",
            "severity":   extra.get("severity") or "info",
            "line":       (r.get("start") or {}).get("line"),
            "end_line":   (r.get("end")   or {}).get("line"),
            "excerpt":    (extra.get("lines") or "")[:4000],
            "message":    extra.get("message") or "",
            "rule_id":    r.get("check_id"),
            "language":   (extra.get("metadata") or {}).get("language"),
            "extra": {
                "fix":      extra.get("fix"),
                "metavars": extra.get("metavars"),
            },
        })
    return out


__all__ = [
    "ExternalScan", "ExternalScanHandle",
    "ingest_findings", "ingest_sarif", "ingest_sandbox_report",
    "parse_semgrep_output",
]
