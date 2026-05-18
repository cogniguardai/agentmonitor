"""
SARIF (Static Analysis Results Interchange Format) parser for AgentMonitor.

SARIF is an OASIS standard JSON format that almost every modern static-
analysis tool supports as an output target: Semgrep, CodeQL, Bandit
(via plugin), Snyk, Trivy, ESLint, Checkov, Gitleaks, ... If a tool
can emit SARIF, AgentMonitor can ingest its findings with NO per-tool
adapter code.

WHY THIS EXISTS
===============

`adapters/findings.py` already gives us a generic ingest surface, but
it expects pre-normalized finding dicts. Each tool has its own native
JSON shape, so without SARIF we'd grow one `parse_X_output()` per tool
(we already have `parse_semgrep_output` and it would only get worse).

SARIF is the standard solution: every tool worth using already emits
it. By parsing SARIF once, the integration burden for adding a new
tool drops to: "run it with --sarif and POST the result." Zero code on
our side. That's what makes the "AgentMonitor is plumbing over
existing scanners" story actually true.

WHAT WE DO NOT DO
=================

This module owns NO detection logic. We translate field names and
SARIF severity levels -> our severity enum, nothing else. The user's
tool already detected the issues; we just persist them in a uniform
shape so the Code Scan and Scanner Obs tabs can render across tools.

We also do NOT enrich, score, rank, or otherwise opine on findings.
A SARIF result reported by the tool as `warning` becomes our `medium`
no matter what the rule is about; we never decide a SQLi warning is
"actually critical" or that a CSP note is "probably exploitable."

SPEC REFERENCE
==============

SARIF v2.1.0  https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html

We target v2.1.0 because that is what every mainstream tool emits in
2025. Older 2.0 documents will mostly parse too -- the fields we read
(`runs[].tool.driver.{name,version,rules}`, `runs[].results[]` with
`ruleId` / `level` / `message` / `locations[].physicalLocation`) have
been stable since 2.0.

USAGE
=====

>>> from agent_monitor.adapters.sarif import parse_sarif
>>> runs = parse_sarif(sarif_json_str_or_dict)
>>> for run in runs:
...     print(run["tool"], run["scanner_version"], len(run["findings"]))

For one-shot ingest, prefer `agent_monitor.adapters.findings.ingest_sarif`
which wires this parser into `ExternalScan` and the Scanner Obs KPIs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

# SARIF defines four `level` values: 'error' / 'warning' / 'note' / 'none'.
# The spec also allows the absence of a level on a result, in which case
# the rule's defaultConfiguration.level applies; if that's missing too,
# the spec says default = 'warning'.
_SARIF_LEVEL_TO_SEVERITY = {
    "error":   "high",
    "warning": "medium",
    "note":    "low",
    "none":    "info",
}


def _security_severity_to_enum(score: float) -> str:
    """CodeQL (and the wider GitHub Advanced Security ecosystem) attaches
    a CVSS-style 0-10 number on `rule.properties['security-severity']`.
    When present it is strictly more informative than the four-level
    enum, so we prefer it.

    Buckets follow the convention used by the GitHub Code Scanning UI:
      9.0+  -> critical
      7.0+  -> high
      4.0+  -> medium
      >0    -> low
      0     -> info
    """
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    if score > 0.0:
        return "low"
    return "info"


# ---------------------------------------------------------------------------
# URI / location helpers
# ---------------------------------------------------------------------------

def _strip_file_scheme(uri: str) -> str:
    """SARIF `artifactLocation.uri` is typically a project-relative
    path, but the spec allows a `file://` URL too. We accept either and
    return a plain path string. Windows file URLs like
    `file:///C:/foo/bar.py` are unwrapped to `C:/foo/bar.py`.
    """
    if not uri:
        return "?"
    if "://" not in uri:
        return uri
    parsed = urlparse(uri)
    path = unquote(parsed.path or "")
    # `/C:/foo` -> `C:/foo` on Windows file URLs
    if len(path) >= 3 and path.startswith("/") and path[2] == ":":
        path = path[1:]
    return path or "?"


def _first_location(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the first physicalLocation from a SARIF result. We use
    the first because (a) that's what every UI does and (b) further
    locations are usually data-flow steps, not the primary site."""
    locs = result.get("locations") or []
    if not locs:
        return {}
    phys = (locs[0].get("physicalLocation") or {})
    art = (phys.get("artifactLocation") or {})
    region = (phys.get("region") or {})
    snippet = (region.get("snippet") or {})
    return {
        "file_path":  _strip_file_scheme(art.get("uri") or ""),
        "start_line": region.get("startLine"),
        "end_line":   region.get("endLine"),
        "snippet":    snippet.get("text") or "",
    }


# ---------------------------------------------------------------------------
# Rule resolution
# ---------------------------------------------------------------------------

def _build_rule_index(
    driver: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Index a run's rule definitions both by `id` AND by position.
    SARIF results reference rules by `ruleId` (string) OR by `ruleIndex`
    (position in `tool.driver.rules`), so we need both lookups.
    """
    rules = driver.get("rules") or []
    by_id: Dict[str, Dict[str, Any]] = {}
    by_index: List[Dict[str, Any]] = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        by_index.append(r)
        rid = r.get("id")
        if rid:
            by_id[rid] = r
    return by_id, by_index


def _resolve_rule(
    result: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
    by_index: List[Dict[str, Any]],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (rule_id, rule_obj) for a single result.

    SARIF lets a result identify its rule three ways:
      1. result['rule']['id']    (full reference, takes precedence)
      2. result['ruleId']        (string id)
      3. result['ruleIndex']     (position in tool.driver.rules)
    We try them in that order, then look the id up in the rule index.
    """
    rule_obj: Dict[str, Any] = {}
    rid: Optional[str] = None
    rule_ref = result.get("rule")
    if isinstance(rule_ref, dict):
        rid = rule_ref.get("id")
    if not rid:
        rid = result.get("ruleId")
    if rid and rid in by_id:
        rule_obj = by_id[rid]
    else:
        idx = result.get("ruleIndex")
        if isinstance(idx, int) and 0 <= idx < len(by_index):
            rule_obj = by_index[idx]
            rid = rid or rule_obj.get("id")
    return rid, rule_obj


def _security_severity_of(rule_obj: Dict[str, Any]) -> Optional[float]:
    props = (rule_obj.get("properties") or {})
    raw = props.get("security-severity")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _severity_for(result: Dict[str, Any], rule_obj: Dict[str, Any]) -> str:
    # 1. CodeQL's security-severity (CVSS 0-10) when present.
    score = _security_severity_of(rule_obj)
    if score is not None:
        return _security_severity_to_enum(score)
    # 2. result.level
    lvl = result.get("level")
    # 3. rule.defaultConfiguration.level (fallback per SARIF spec)
    if not lvl:
        lvl = (rule_obj.get("defaultConfiguration") or {}).get("level")
    return _SARIF_LEVEL_TO_SEVERITY.get((lvl or "").lower(), "info")


def _kind_for(rule_id: Optional[str], rule_obj: Dict[str, Any]) -> str:
    """Pick a short, stable category label for the `kind` column.

    We prefer the rule id (or its last dotted segment) because it's
    stable across scanner runs; the human-readable `name` often changes
    between rule-pack versions and would break drift detection.

    Examples:
      python.lang.security.audit.sqli  -> 'sqli'
      py/sql-injection                 -> 'py/sql-injection'   (no dots)
      B608                             -> 'B608'
    """
    if rule_id:
        if "." in rule_id:
            tail = rule_id.rsplit(".", 1)[-1]
            if tail:
                return tail
        return rule_id
    name = rule_obj.get("name") if isinstance(rule_obj, dict) else None
    return name or "external"


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _message_text(result: Dict[str, Any]) -> str:
    """SARIF messages are objects with text/markdown variants. We
    prefer plain text. Tool messages with placeholder substitution
    (`{0}`, `{1}`...) are returned verbatim -- substitution requires
    `arguments` and is rare in scanner output."""
    msg = result.get("message")
    if isinstance(msg, dict):
        return (msg.get("text") or msg.get("markdown") or "").strip()
    if isinstance(msg, str):
        return msg
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sarif(payload: Any) -> List[Dict[str, Any]]:
    """Parse a SARIF document into one entry per `runs[]` item.

    Args:
        payload: Either a SARIF document as a dict, or a JSON string.

    Returns:
        A list of normalized runs. Each entry is:

            {
              "tool":            "<lowercased tool.driver.name>",
              "scanner_version": "<tool.driver.semanticVersion | .version | None>",
              "findings":        [ <ingest_findings-compatible dict>, ... ],
              "n_results":       <int>,   # original SARIF results count
            }

        The `findings` entries are shaped like
        `agent_monitor.adapters.findings.ingest_findings` expects, so
        the parser can feed that function directly.

    Raises:
        ValueError: if `payload` is not a JSON object / parseable JSON,
            or doesn't look like SARIF (no `runs` key).
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError("SARIF payload must be a JSON object")
    if "runs" not in payload:
        raise ValueError(
            "missing 'runs' key -- does this look like a SARIF document? "
            f"top-level keys: {sorted(payload.keys())[:8]}"
        )
    runs_in = payload.get("runs") or []
    if not isinstance(runs_in, list):
        raise ValueError("'runs' must be a list")

    out: List[Dict[str, Any]] = []
    for run in runs_in:
        if not isinstance(run, dict):
            continue
        tool = run.get("tool") or {}
        driver = tool.get("driver") or {}
        tool_name = (driver.get("name") or "sarif").strip().lower() or "sarif"
        version = driver.get("semanticVersion") or driver.get("version")

        by_id, by_index = _build_rule_index(driver)

        results = run.get("results") or []
        findings: List[Dict[str, Any]] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            rid, rule_obj = _resolve_rule(r, by_id, by_index)
            loc = _first_location(r)
            sec_sev = _security_severity_of(rule_obj)
            findings.append({
                "file_path":  loc.get("file_path") or "?",
                "kind":       _kind_for(rid, rule_obj),
                "severity":   _severity_for(r, rule_obj),
                "line":       loc.get("start_line"),
                "end_line":   loc.get("end_line"),
                "excerpt":    loc.get("snippet") or "",
                "message":    _message_text(r),
                "rule_id":    rid,
                # Extra metadata kept for the chunk_summary blob. NOT
                # used for detection -- purely audit/breadcrumb info.
                "extra": {
                    "sarif_level":         r.get("level"),
                    "security_severity":   sec_sev,
                    "partialFingerprints": r.get("partialFingerprints"),
                    "fingerprints":        r.get("fingerprints"),
                    "tags":                (rule_obj.get("properties") or {}).get("tags"),
                },
            })

        out.append({
            "tool":            tool_name,
            "scanner_version": version,
            "findings":        findings,
            "n_results":       len(results),
        })

    return out


__all__ = ["parse_sarif"]
