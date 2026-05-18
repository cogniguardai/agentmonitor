"""
External VM/sandbox-report parser for AgentMonitor (v1.10).

Sister module to ``agent_monitor.adapters.sarif``. Where ``sarif.py``
ingests *static* analyzer output (Semgrep / CodeQL / Bandit / ...),
this module ingests *dynamic* analyzer output -- the JSON reports
produced by full-system sandboxes that detonate a sample in an
isolated VM and observe its behavior.

WHY THIS EXISTS
===============

Security teams that already use a sandbox like Cuckoo, Joe Sandbox,
ANY.RUN or VMRay typically have those reports stockpiled as JSON files.
Until now there was no way to push that information into AgentMonitor
without writing a per-sandbox adapter from scratch. This module is the
"AgentMonitor is plumbing over existing sandboxes" piece -- pick a
format, normalize to our `code_finding` schema, get triage / drift /
KPI tracking for free.

WHAT WE DO
==========

* Parse Cuckoo Sandbox 2.x JSON reports (the open-source de-facto
  research format).
* Parse a small "generic sandbox" envelope that any tool can target if
  it doesn't natively emit Cuckoo's shape -- Joe / VMRay / ANY.RUN
  exports can be normalized to it in <30 lines.
* Map sandbox `signatures` / `signals` -> AgentMonitor `code_finding`
  rows. Each signature becomes one finding; the analyzed sample's
  SHA-256 (or its filename if no SHA is present) is stored in
  `file_path` so the existing UI works without changes.

WHAT WE DO NOT DO
=================

* We do NOT run the sandbox. The user already detonated their sample
  and produced a report; we're just persisting it.
* We do NOT decide whether a behavior is malicious. The sandbox's own
  severity / score / signature list is taken at face value. A Cuckoo
  signature with severity=3 maps to our `medium`, full stop -- we
  never re-grade based on rule names or descriptions.
* We do NOT enrich with threat intel, IOC reputation, MITRE walks,
  or anything that turns "sandbox observation" into "exploit
  candidate". Those lines live in the AgentMonitor scope doc.
* We do NOT model time-ordered traces. AgentMonitor's per-finding
  schema is one event per row; for an MVP that's fine -- we surface
  WHICH signatures fired, not the temporal sequence of API calls.
  Adding a `sandbox_event` time-series table is a future extension.

INPUT FORMATS
=============

1. **Cuckoo Sandbox 2.x JSON** (auto-detected when the document has
   ``signatures`` AND (``info`` OR ``target``)). Schema reference:
   https://cuckoo.readthedocs.io/en/latest/usage/api/#tasks-report
   We read:
     - ``info.version``                       -> scanner_version
     - ``info.id``                            -> scan label
     - ``target.file.{sha256, name, type}``   -> file_path / extra
     - ``signatures[].{name, description,
                       severity, categories,
                       references, marks}``   -> one finding each

2. **Generic sandbox envelope** (auto-detected when the document has
   a top-level ``signals`` array). The shape is::

       {
         "tool":     "joesandbox",          # required, lowercased
         "version":  "30.0.4",              # optional
         "sample":   {"name": "x.exe",
                      "sha256": "abc..."},  # at least one of name/sha256
         "verdict":  {"severity": "high",
                      "score": 8.5},        # optional rollup
         "signals":  [
             {
               "id":          "T1059.001",
               "name":        "PowerShell",
               "category":    "execution",
               "severity":    "high",
               "description": "...",
               "evidence":    "Long verbatim quote from the report",
               "mitre":       ["T1059.001"]
             },
             ...
         ]
       }

   This is the format we recommend partners and one-off scripts target
   when they want to push non-Cuckoo data through AgentMonitor without
   us writing yet another per-vendor parser.

USAGE
=====

>>> from agent_monitor.adapters.sandbox_report import parse_sandbox_report
>>> result = parse_sandbox_report(report_json_str_or_dict)
>>> result["tool"], result["scanner_version"], len(result["findings"])

For one-shot ingest, prefer
``agent_monitor.adapters.findings.ingest_sandbox_report`` which wires
this parser into ``ExternalScan`` and the Scanner Obs KPIs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------

# Cuckoo signatures use a 1..5 integer severity. The mapping below is
# the one Cuckoo's own web UI uses (informational/low/medium/high/critical),
# so a user comparing AgentMonitor against the native Cuckoo dashboard
# sees the same colour for the same signature.
_CUCKOO_SEVERITY_TO_ENUM = {
    1: "info",
    2: "low",
    3: "medium",
    4: "high",
    5: "critical",
}

# Allowed AgentMonitor severity values -- anything else is normalized
# down to "info" so we never invent a higher grade than the report
# actually said.
_VALID_SEVERITY = ("info", "low", "medium", "high", "critical")


def _coerce_severity(raw: Any) -> str:
    """Map a sandbox-supplied severity to our enum. Accepts:
      * one of our enum strings (case-insensitive)
      * Cuckoo's 1..5 integer
      * a 0..10 float (CVSS-style); same buckets as SARIF's
        security-severity numeric.
      * common variants ('warning', 'malicious', 'suspicious', ...)
    Anything unrecognized -> "info" (we never invent severity).
    """
    if raw is None:
        return "info"
    # Numeric: ints 1..5 are Cuckoo, otherwise treat as CVSS 0..10.
    if isinstance(raw, bool):
        return "info"  # bool is an int subclass; ignore explicitly
    if isinstance(raw, int):
        if 1 <= raw <= 5:
            return _CUCKOO_SEVERITY_TO_ENUM[raw]
        return _coerce_severity(float(raw))
    if isinstance(raw, float):
        if raw >= 9.0:
            return "critical"
        if raw >= 7.0:
            return "high"
        if raw >= 4.0:
            return "medium"
        if raw > 0.0:
            return "low"
        return "info"
    s = str(raw).strip().lower()
    if s in _VALID_SEVERITY:
        return s
    return {
        "informational": "info",
        "warning":       "medium",
        "warn":          "medium",
        "moderate":      "medium",
        "error":         "high",
        "severe":        "high",
        "malicious":     "high",
        "suspicious":    "medium",
        "clean":         "info",
        "benign":        "info",
        "minor":         "low",
        "major":         "medium",
        "blocker":       "critical",
    }.get(s, "info")


# ---------------------------------------------------------------------------
# Sample-identifier extraction
# ---------------------------------------------------------------------------

def _sample_id(sample: Dict[str, Any]) -> str:
    """Pick a stable identifier for the analyzed sample. We prefer the
    SHA-256 because it's content-addressed and stable across submissions
    of the same file; fall back to filename, then to '?'.

    The chosen value lives in `code_finding.file_path` so the existing
    Code Scan / Scanner Obs UIs (which group by file_path) work without
    any changes -- a sandbox "file" is conceptually one sample.
    """
    if not isinstance(sample, dict):
        return "?"
    sha = sample.get("sha256") or sample.get("sha-256") or sample.get("sha_256")
    if isinstance(sha, str) and len(sha) >= 32:
        return f"sha256:{sha.lower()}"
    name = sample.get("name") or sample.get("filename") or sample.get("file_name")
    if isinstance(name, str) and name:
        return name
    return "?"


# ---------------------------------------------------------------------------
# Cuckoo Sandbox 2.x parser
# ---------------------------------------------------------------------------

def _looks_like_cuckoo(payload: Dict[str, Any]) -> bool:
    """Cuckoo reports always have a top-level `signatures` list and at
    least one of `info` / `target` / `behavior`. We use the conjunction
    so a generic envelope that happens to have a `signatures` key
    doesn't get misclassified as Cuckoo."""
    if not isinstance(payload, dict):
        return False
    if "signatures" not in payload:
        return False
    return any(k in payload for k in ("info", "target", "behavior"))


def _cuckoo_first_mark_excerpt(marks: Any) -> str:
    """Cuckoo signatures attach a list of `marks` -- evidence items
    that triggered the signature. They come in many shapes (call,
    file, registry, generic). We render the FIRST one as a one-line
    excerpt, because the UI displays one excerpt per finding and the
    first mark is generally the most representative.

    We are deliberately conservative: long structured payloads are
    truncated. We never invoke any kind of pretty-printer that might
    re-format strings -- whatever the report said, we preserve.
    """
    if not isinstance(marks, list) or not marks:
        return ""
    m = marks[0]
    if not isinstance(m, dict):
        return str(m)[:400]
    mtype = m.get("type") or "mark"
    # Common mark shapes (Cuckoo internal types):
    if mtype == "call":
        call = m.get("call") or {}
        if isinstance(call, dict):
            api = call.get("api") or "?"
            args = call.get("arguments") or {}
            return f"call {api}({json.dumps(args, default=str)[:300]})"
    if mtype in ("file", "registry"):
        ioc = m.get("ioc") or m.get("value") or m.get(mtype) or ""
        return f"{mtype}: {ioc}"
    if mtype == "generic":
        # "generic" marks carry arbitrary kv pairs; pick a description.
        for k in ("description", "text", "value"):
            if k in m:
                return f"{m[k]}"[:400]
    # Fallback: dump the mark as compact JSON.
    return json.dumps(m, default=str)[:400]


def _parse_cuckoo(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Cuckoo 2.x report. Returns one normalized scan dict
    (Cuckoo reports describe a single sample, so unlike SARIF there is
    no list of runs -- the parser always returns ONE entry)."""
    info     = payload.get("info")     if isinstance(payload.get("info"),     dict) else {}
    target   = payload.get("target")   if isinstance(payload.get("target"),   dict) else {}
    sample   = target.get("file")      if isinstance(target.get("file"),      dict) else {}

    file_path = _sample_id(sample)
    scanner_version = (info.get("version") or "").strip() or None
    task_id = info.get("id")

    sigs = payload.get("signatures") or []
    findings: List[Dict[str, Any]] = []
    for s in sigs:
        if not isinstance(s, dict):
            continue
        name        = (s.get("name") or "signature").strip() or "signature"
        description = (s.get("description") or "").strip()
        severity    = _coerce_severity(s.get("severity"))
        categories  = s.get("categories") or []
        marks       = s.get("marks") or []
        refs        = s.get("references") or []
        excerpt     = _cuckoo_first_mark_excerpt(marks) or description

        findings.append({
            "file_path":  file_path,
            "kind":       name,                # signature id is the category label
            "severity":   severity,
            "line":       None,                # N/A: dynamic-analysis events have no source line
            "end_line":   None,
            "excerpt":    excerpt[:4000],
            "message":    description,
            "rule_id":    name,
            "extra": {
                "categories":     categories if isinstance(categories, list) else [categories],
                "references":     refs if isinstance(refs, list) else [],
                "n_marks":        len(marks) if isinstance(marks, list) else 0,
                "cuckoo_severity": s.get("severity"),
                "task_id":        task_id,
                "sample_name":    sample.get("name") if isinstance(sample, dict) else None,
                "sample_type":    sample.get("type") if isinstance(sample, dict) else None,
            },
        })

    return {
        "tool":            "cuckoo",
        "scanner_version": scanner_version,
        "sample_id":       file_path,
        "findings":        findings,
        "n_signals":       len(findings),
    }


# ---------------------------------------------------------------------------
# Generic sandbox envelope parser
# ---------------------------------------------------------------------------

def _looks_like_generic(payload: Dict[str, Any]) -> bool:
    """Our generic envelope is recognized by a top-level `signals`
    array. We accept it even without a `tool` field (defaults to
    "sandbox"), but the usual case is the partner explicitly tags it.
    """
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("signals"), list)


def _parse_generic(payload: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = (payload.get("tool") or "sandbox").strip().lower() or "sandbox"
    scanner_version = payload.get("version")
    if scanner_version is not None:
        scanner_version = str(scanner_version).strip() or None
    sample = payload.get("sample") if isinstance(payload.get("sample"), dict) else {}
    file_path = _sample_id(sample)
    verdict  = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}

    signals = payload.get("signals") or []
    findings: List[Dict[str, Any]] = []
    for s in signals:
        if not isinstance(s, dict):
            continue
        sid = s.get("id") or s.get("name") or "signal"
        name = s.get("name") or sid
        category = s.get("category") or s.get("kind") or "signal"
        severity = _coerce_severity(s.get("severity") or s.get("score"))
        evidence = s.get("evidence") or s.get("excerpt") or ""
        description = s.get("description") or s.get("message") or name
        mitre = s.get("mitre") or s.get("attack")
        if isinstance(mitre, str):
            mitre = [mitre]

        findings.append({
            "file_path":  file_path,
            "kind":       str(category)[:80],
            "severity":   severity,
            "line":       None,
            "end_line":   None,
            "excerpt":    str(evidence)[:4000] or str(description)[:4000],
            "message":    str(description)[:2000],
            "rule_id":    str(sid)[:80],
            "extra": {
                "name":      str(name),
                "mitre":     mitre if isinstance(mitre, list) else None,
                "category":  str(category),
                "score":     s.get("score"),
                "severity_in": s.get("severity"),
                "verdict":   verdict or None,
                "sample_name":   sample.get("name") if isinstance(sample, dict) else None,
                "sample_sha256": sample.get("sha256") if isinstance(sample, dict) else None,
            },
        })

    return {
        "tool":            tool_name,
        "scanner_version": scanner_version,
        "sample_id":       file_path,
        "findings":        findings,
        "n_signals":       len(findings),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sandbox_report(
    payload: Any, *, format: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a sandbox report into normalized AgentMonitor findings.

    Args:
        payload: a sandbox JSON document, either as a dict or a raw
                 JSON string.
        format:  optional explicit format hint -- one of
                 ``"cuckoo"`` / ``"generic"``. When omitted, we
                 auto-detect: a document with both ``signatures`` and
                 (``info`` OR ``target`` OR ``behavior``) is treated
                 as Cuckoo; a document with a top-level ``signals``
                 array is treated as the generic envelope.

    Returns:
        A dict shaped like::

            {
              "tool":            "<tool name, lowercased>",
              "scanner_version": "<version string or None>",
              "sample_id":       "<sha256:... or filename>",
              "findings":        [ <ingest_findings-compatible dicts> ],
              "n_signals":       <int>,   # original signature/signal count
            }

        The `findings` entries match the shape that
        ``agent_monitor.adapters.findings.ingest_findings`` expects, so
        the parser feeds directly into that pipeline.

    Raises:
        ValueError: if ``payload`` is not parseable JSON / not a JSON
            object, or if no recognized format is detected.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError("sandbox report payload must be a JSON object")

    fmt = (format or "").strip().lower() or None
    if fmt is None:
        if _looks_like_cuckoo(payload):
            fmt = "cuckoo"
        elif _looks_like_generic(payload):
            fmt = "generic"
        else:
            raise ValueError(
                "could not auto-detect sandbox report format: "
                "expected either Cuckoo (top-level 'signatures' + "
                "'info'/'target'/'behavior') or generic ('signals' "
                f"array). top-level keys: {sorted(payload.keys())[:8]}"
            )

    if fmt == "cuckoo":
        return _parse_cuckoo(payload)
    if fmt == "generic":
        return _parse_generic(payload)
    raise ValueError(
        f"unsupported sandbox report format: {fmt!r} "
        "(expected 'cuckoo' or 'generic')"
    )


__all__ = ["parse_sandbox_report"]
