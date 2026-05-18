"""
Offensive-pattern trace classifier (v1.8).

WHAT THIS IS
============

A *defender-side* pattern matcher that reads the TEXT of an agent's
`trace_event` rows and flags signals associated with exploit-development
or offensive-security workflows. It is the AgentMonitor analogue of a
YARA rule for agent activity: a fingerprint of behaviour, not a copy of
the behaviour itself.

WHAT THIS IS NOT
================

* NOT a detection rule generator. We do not write rules tuned to find
  specific bug classes.
* NOT a vulnerability scanner. We do not look at source code, driver
  binaries, decompiler output, or anything outside the user's own
  trace text.
* NOT a verdict. The score is a coarse confidence number. Surfacing it
  in the dashboard is to *prompt review*, not to declare an agent
  hostile. Every match is shown, with its source URL, so the user can
  see exactly why a number was assigned.

SIGNATURE PROVENANCE
====================

Every signature below cites a public defender-side source:

  * MITRE ATT&CK technique IDs (https://attack.mitre.org)
  * Microsoft public WDK / Windows kernel API documentation
    (https://learn.microsoft.com/windows-hardware/drivers/)
  * LOLDrivers project (https://www.loldrivers.io) -- a community
    catalogue of *known-abused* drivers, post-disclosure data
  * Microsoft Vulnerable Driver Blocklist
    (https://learn.microsoft.com/windows/security/...)
  * Generic exploit-dev terminology common in defender literature

We do NOT include patterns that constitute novel detection content.
Every signature here is the kind of string that already appears in
public threat reports, defender training materials, or vendor docs.

WEIGHTS
=======

All weights are small (0.05 -- 0.20). A run scoring 0.5+ has
accumulated multiple independent signals, which is the signal the user
should care about: not a single API mention, but a *cluster* of them.
The score is capped at 1.0.

LIMITATIONS
===========

* Pattern matching has well-known false positives. A defender writing
  detection logic against AgentMonitor logs may legitimately mention
  `MmMapIoSpace` or `T1068` -- the classifier will flag their run.
  This is acceptable: the dashboard shows hits, not verdicts, so the
  user can confirm.

* The classifier has no opinion on whether the agent is "good" or
  "bad". It reports patterns. Interpretation is the user's job.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent_monitor import db


# ---------------------------------------------------------------------------
# Signature dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signature:
    id:          str       # short stable id, e.g. "ke.api.MmMapIoSpace"
    domain:      str       # one of the DOMAIN_* below
    weight:      float     # 0.05 -- 0.20
    pattern:     str       # plain string OR regex; see `regex` field
    regex:       bool      # if False, pattern is matched as case-insensitive substring
    description: str       # one-line, defender-readable
    source:      str       # public URL or "common defender literature"


# Domains -- used to decide the run's dominant `classifier_kind`
DOMAIN_RE_TOOLING       = "re_tooling"        # IDA / Ghidra / radare2 fingerprints
DOMAIN_KERNEL_API       = "kernel_api"        # Windows kernel API names
DOMAIN_BYOVD            = "byovd"             # vulnerable / abused driver indicators
DOMAIN_EXPLOIT_LEXICON  = "exploit_lexicon"   # generic exploit-dev vocabulary
DOMAIN_ATTACK_TECHNIQUE = "attack_technique"  # MITRE ATT&CK technique IDs
DOMAIN_RECON            = "recon"             # batched-corpus / mass-scan patterns


# ---------------------------------------------------------------------------
# Signature library (small, public, conservative)
# ---------------------------------------------------------------------------

# Matched as case-insensitive substrings unless `regex=True`.
SIGNATURES: List[Signature] = [
    # --- RE TOOLING --------------------------------------------------------
    Signature("re.tool.ida",        DOMAIN_RE_TOOLING, 0.10,
              r"\b(ida\s*pro|idapython|idc\.|idaapi|ida_kernwin)\b", True,
              "IDA Pro / IDAPython tooling reference",
              "https://hex-rays.com/ida-pro/"),
    Signature("re.tool.ghidra",     DOMAIN_RE_TOOLING, 0.10,
              r"\bghidra\b", True,
              "Ghidra reverse engineering platform",
              "https://ghidra-sre.org/"),
    Signature("re.tool.radare2",    DOMAIN_RE_TOOLING, 0.10,
              r"\b(radare2|r2pipe|rabin2)\b", True,
              "radare2 reverse engineering toolkit",
              "https://rada.re/"),
    Signature("re.tool.binaryninja",DOMAIN_RE_TOOLING, 0.10,
              r"\bbinary\s*ninja\b", True,
              "Binary Ninja reverse engineering platform",
              "https://binary.ninja/"),
    Signature("re.tool.dumpbin",    DOMAIN_RE_TOOLING, 0.05,
              r"\bdumpbin\s+/(disasm|exports|imports)\b", True,
              "Windows dumpbin disassembly invocation",
              "https://learn.microsoft.com/cpp/build/reference/dumpbin-options"),

    # --- WINDOWS KERNEL API NAMES (public WDK docs) ------------------------
    # The mere mention is not a detection -- many of these appear in
    # benign driver development. Weights are deliberately low.
    Signature("ke.api.MmMapIoSpace",     DOMAIN_KERNEL_API, 0.15,
              "MmMapIoSpace", False,
              "MmMapIoSpace -- maps physical to virtual address (kernel API)",
              "https://learn.microsoft.com/windows-hardware/drivers/ddi/wdm/nf-wdm-mmmapiospace"),
    Signature("ke.api.ZwMapViewOfSection", DOMAIN_KERNEL_API, 0.10,
              "ZwMapViewOfSection", False,
              "ZwMapViewOfSection -- maps a section into a process",
              "https://learn.microsoft.com/windows-hardware/drivers/ddi/wdm/nf-wdm-zwmapviewofsection"),
    Signature("ke.api.KeStackAttach",    DOMAIN_KERNEL_API, 0.10,
              "KeStackAttachProcess", False,
              "KeStackAttachProcess -- attach to another process's address space",
              "https://learn.microsoft.com/windows-hardware/drivers/ddi/ntddk/nf-ntddk-kestackattachprocess"),
    Signature("ke.api.PsLookupProcess",  DOMAIN_KERNEL_API, 0.08,
              "PsLookupProcessByProcessId", False,
              "PsLookupProcessByProcessId -- find an EPROCESS by PID",
              "https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/nf-ntifs-pslookupprocessbyprocessid"),
    Signature("ke.api.HalGetBusData",    DOMAIN_KERNEL_API, 0.10,
              "HalGetBusData", False,
              "HalGetBusData -- legacy bus access, often used in vulnerable drivers",
              "https://learn.microsoft.com/windows-hardware/drivers/ddi/ntddk/nf-ntddk-halgetbusdata"),
    Signature("ke.api.IoCreateSymlink",  DOMAIN_KERNEL_API, 0.05,
              "IoCreateSymbolicLink", False,
              "IoCreateSymbolicLink -- driver namespace exposure",
              "https://learn.microsoft.com/windows-hardware/drivers/ddi/wdm/nf-wdm-iocreatesymboliclink"),

    # --- BYOVD / DRIVER ABUSE INDICATORS -----------------------------------
    Signature("byovd.term",          DOMAIN_BYOVD, 0.20,
              r"\bBYOVD\b|bring\s+your\s+own\s+vulnerable\s+driver", True,
              "BYOVD (Bring Your Own Vulnerable Driver) terminology",
              "https://www.loldrivers.io/"),
    Signature("byovd.loldrivers",    DOMAIN_BYOVD, 0.15,
              r"\bloldrivers\b|living[- ]off[- ]the[- ]land\s+driver", True,
              "Reference to LOLDrivers / living-off-the-land drivers",
              "https://www.loldrivers.io/"),
    Signature("byovd.whql",          DOMAIN_BYOVD, 0.08,
              r"\bWHQL\b", True,
              "WHQL (Windows Hardware Quality Labs) signing context",
              "https://learn.microsoft.com/windows-hardware/drivers/develop/whql-release-signature"),
    Signature("byovd.signed_kernel", DOMAIN_BYOVD, 0.10,
              r"signed[- ]kernel[- ]driver|driver[- ]signing[- ]bypass", True,
              "Signed-kernel-driver / driver-signing-bypass language",
              "https://learn.microsoft.com/windows-hardware/drivers/install/kernel-mode-code-signing-policy"),
    Signature("byovd.driver_blocklist", DOMAIN_BYOVD, 0.10,
              r"vulnerable\s+driver\s+block(list|\s+list)|HVCI\s+block", True,
              "Microsoft Vulnerable Driver Blocklist / HVCI-block context",
              "https://learn.microsoft.com/windows/security/threat-protection/windows-defender-application-control/microsoft-recommended-driver-block-rules"),
    # A few well-known abused driver names from the LOLDrivers public catalogue.
    # These are post-disclosure, defender data.
    Signature("byovd.driver.rtcore", DOMAIN_BYOVD, 0.15,
              r"\brtcore64\.sys\b", True,
              "RTCore64.sys -- catalogued vulnerable driver (LOLDrivers)",
              "https://www.loldrivers.io/drivers/RTCore64/"),
    Signature("byovd.driver.capcom", DOMAIN_BYOVD, 0.15,
              r"\bcapcom\.sys\b", True,
              "Capcom.sys -- catalogued vulnerable driver (LOLDrivers)",
              "https://www.loldrivers.io/drivers/Capcom/"),
    Signature("byovd.driver.gdrv",   DOMAIN_BYOVD, 0.15,
              r"\bgdrv\.sys\b", True,
              "gdrv.sys -- catalogued vulnerable driver (LOLDrivers)",
              "https://www.loldrivers.io/drivers/gdrv/"),

    # --- GENERIC EXPLOIT-DEV LEXICON ---------------------------------------
    Signature("xd.term.rop",         DOMAIN_EXPLOIT_LEXICON, 0.10,
              r"\bROP\s+(gadget|chain)\b|return[- ]oriented\s+programming", True,
              "ROP-chain / gadget-construction language",
              "common defender literature"),
    Signature("xd.term.shellcode",   DOMAIN_EXPLOIT_LEXICON, 0.08,
              r"\bshellcode\b", True,
              "Shellcode terminology",
              "common defender literature"),
    Signature("xd.term.smep",        DOMAIN_EXPLOIT_LEXICON, 0.10,
              r"\bSMEP\b|\bSMAP\b", True,
              "SMEP / SMAP kernel mitigation references",
              "https://en.wikipedia.org/wiki/Supervisor_Mode_Access_Prevention"),
    Signature("xd.term.kaslr",       DOMAIN_EXPLOIT_LEXICON, 0.08,
              r"\bk?ASLR\s+(bypass|leak)\b", True,
              "ASLR/KASLR-bypass language",
              "common defender literature"),
    Signature("xd.term.primitives",  DOMAIN_EXPLOIT_LEXICON, 0.12,
              r"(arbitrary|kernel)\s+(read|write|read[/]write|R\s*/\s*W)\s+primitive", True,
              "Arbitrary kernel R/W primitive language",
              "common defender literature"),
    Signature("xd.term.exploit_score",DOMAIN_EXPLOIT_LEXICON, 0.15,
              r"score.{0,40}exploit(abilit(y|y)|able)", True,
              "'score X for exploitability' framing",
              "common defender literature"),
    Signature("xd.term.privesc",     DOMAIN_EXPLOIT_LEXICON, 0.08,
              r"\bprivilege\s+escalation\b|\bprivesc\b|\blpe\b", True,
              "Privilege escalation / LPE language",
              "common defender literature"),

    # --- MITRE ATT&CK TECHNIQUE IDS ----------------------------------------
    Signature("mitre.T1068",  DOMAIN_ATTACK_TECHNIQUE, 0.08,
              r"\bT1068\b", True,
              "ATT&CK T1068 -- Exploitation for Privilege Escalation",
              "https://attack.mitre.org/techniques/T1068/"),
    Signature("mitre.T1543",  DOMAIN_ATTACK_TECHNIQUE, 0.06,
              r"\bT1543(\.\d+)?\b", True,
              "ATT&CK T1543 -- Create or Modify System Process",
              "https://attack.mitre.org/techniques/T1543/"),
    Signature("mitre.T1014",  DOMAIN_ATTACK_TECHNIQUE, 0.08,
              r"\bT1014\b", True,
              "ATT&CK T1014 -- Rootkit",
              "https://attack.mitre.org/techniques/T1014/"),
    Signature("mitre.T1547",  DOMAIN_ATTACK_TECHNIQUE, 0.05,
              r"\bT1547(\.\d+)?\b", True,
              "ATT&CK T1547 -- Boot or Logon Autostart Execution",
              "https://attack.mitre.org/techniques/T1547/"),

    # --- RECON / MASS-EVAL PATTERNS ----------------------------------------
    Signature("recon.driver_corpus", DOMAIN_RECON, 0.15,
              r"\.sys[\s\"',;]|\\\\system32\\\\drivers\\\\", True,
              "Reference to a corpus of .sys / driver paths",
              "common defender literature"),
    Signature("recon.batch_score",   DOMAIN_RECON, 0.10,
              r"score.{0,40}(0[- ]100|0\s*to\s*100)|exploitabilit(y|y)\s+score", True,
              "Batched 0-100 exploitability scoring patterns",
              "common defender literature"),
]


# Compile patterns once at module load
_COMPILED: List[Tuple[Signature, re.Pattern]] = [
    (s, re.compile(s.pattern, re.IGNORECASE)) if s.regex
    else (s, re.compile(re.escape(s.pattern), re.IGNORECASE))
    for s in SIGNATURES
]


# Hard cap on score so a noisy run doesn't blow past 1.0.
SCORE_CAP = 1.0
# Below this score the run is considered "no signal" and we leave the
# DB columns NULL rather than persisting noise.
NULL_BELOW = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(text: str) -> Dict[str, Any]:
    """Run the signature library against an arbitrary text blob.

    Returns:
        {"score": float in [0,1] (0 if no signals, capped at 1.0),
         "kind":  dominant domain string OR None when score < NULL_BELOW,
         "signals": [{"id", "domain", "weight", "description", "source",
                      "matched_text"}]}

    The function never raises on user input.
    """
    if not text:
        return {"score": 0.0, "kind": None, "signals": []}
    hits: List[Dict[str, Any]] = []
    domain_weight: Dict[str, float] = {}
    raw = 0.0
    for sig, rx in _COMPILED:
        m = rx.search(text)
        if m is None:
            continue
        raw += sig.weight
        domain_weight[sig.domain] = domain_weight.get(sig.domain, 0.0) + sig.weight
        hits.append({
            "id":           sig.id,
            "domain":       sig.domain,
            "weight":       sig.weight,
            "description":  sig.description,
            "source":       sig.source,
            "matched_text": m.group(0)[:200],
        })
    score = min(raw, SCORE_CAP)
    kind: Optional[str] = None
    if score >= NULL_BELOW and domain_weight:
        kind = max(domain_weight.items(), key=lambda kv: kv[1])[0]
    return {"score": round(score, 4), "kind": kind, "signals": hits}


def trace_text_for_run(run_id: int, max_chars: int = 200_000) -> str:
    """Concatenate the text content of all `trace_event` rows for a run
    (plus the run's own input/output) into one string for classification.

    We deliberately read JSON payloads as text -- the classifier is
    string-matching, so we don't care about structure. Capped at
    `max_chars` to avoid pathological cases.
    """
    parts: List[str] = []
    with db.session() as conn:
        r = conn.execute(
            "SELECT input_text, output_text FROM run WHERE id = ?",
            (run_id,)
        ).fetchone()
        if r:
            parts.append(r["input_text"] or "")
            parts.append(r["output_text"] or "")
        rows = conn.execute(
            "SELECT payload_json FROM trace_event WHERE run_id = ? "
            "ORDER BY seq", (run_id,)
        ).fetchall()
        for row in rows:
            parts.append(row["payload_json"] or "")
            if sum(len(p) for p in parts) > max_chars:
                break
    blob = "\n".join(parts)
    return blob[:max_chars]


def classify_run(run_id: int) -> Dict[str, Any]:
    """Classify a single run by id. Returns the same shape as `classify`."""
    text = trace_text_for_run(run_id)
    return classify(text)


def list_signatures() -> List[Dict[str, Any]]:
    """For the UI: list every active signature with its weight + source."""
    return [
        {
            "id":          s.id,
            "domain":      s.domain,
            "weight":      s.weight,
            "description": s.description,
            "source":      s.source,
        }
        for s in SIGNATURES
    ]


__all__ = [
    "classify", "classify_run", "trace_text_for_run", "list_signatures",
    "SIGNATURES", "Signature",
    "DOMAIN_RE_TOOLING", "DOMAIN_KERNEL_API", "DOMAIN_BYOVD",
    "DOMAIN_EXPLOIT_LEXICON", "DOMAIN_ATTACK_TECHNIQUE", "DOMAIN_RECON",
]
