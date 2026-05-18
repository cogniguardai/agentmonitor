"""
agent_monitor.code_scan -- LLM-driven code screening (v1.5).

WHAT THIS IS (and what it isn't)
================================

This module walks a directory of source code and asks a local LLM
(via Ollama) to flag suspicious patterns. Every flagged pattern must be
backed by a verbatim quote from the source (same hallucination defense as
the NLA `notes` machinery).

This is a SCREENING TOOL. Calibrate expectations:

  * It will produce false positives. Triage every finding by hand.
  * It will miss bugs. It is line-window-local; it cannot see cross-file
    call graphs, build scripts, binary blobs, or test harnesses. A real
    supply-chain attack like xz-utils (2024) needed all of those.
  * It is not a replacement for CodeQL, Coverity, Semgrep, syzkaller,
    AFL++, or formal methods. Use those for ground truth. Use this to
    point a human at things to look at first.

Design
------
1. Walk a directory. Filter to a configured set of source extensions.
   Skip files > max_bytes (default 256 KB) -- they are usually generated.
2. Chunk each file into overlapping line windows
   (`max_chunk_lines`, default 200; `overlap_lines`, default 30 -- tuned
   for the small 3B coder model; a larger model can take bigger chunks).
   Why overlap: a bug pattern straddling a chunk boundary should be
   visible to at least one of the two chunks that cover the boundary.
3. For each chunk, call `nla_client.decode_code(...)` synchronously on the
   scan thread. The decode call goes through the existing cache, so
   re-scanning the same file is free.
4. For each finding, persist a row in `code_finding`. Periodically update
   the parent `code_scan` row with progress.
5. The whole walk runs in a daemon thread per scan. The HTTP layer polls
   `/api/scan/{id}` for progress.

We deliberately do NOT use the existing `nla_worker` queue, because:
  - Code scans want one-at-a-time-per-Ollama-instance pacing (the
    worker's drop-oldest backpressure is wrong for a scan: we want
    *every* chunk processed).
  - Code scans need per-scan progress accounting, which doesn't map onto
    a shared global queue.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent_monitor import db, nla_client


# ---------------------------------------------------------------------------
# Configuration: what we treat as source, what we skip
# ---------------------------------------------------------------------------

# Extension -> language label fed to the prompt.
# Conservative list: things where a small coder model has any chance of
# being useful. Expand cautiously.
DEFAULT_EXTENSIONS: Dict[str, str] = {
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell", ".bash": "shell",
    ".sql": "sql",
    ".asm": "asm", ".s": "asm",
}

# Always skip these directory names anywhere in the tree.
DEFAULT_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "venv", ".venv", "env", "__pycache__",
    "target", "build", "dist", "out", "obj",
    ".cache", ".idea", ".vscode",
}

# Defaults you can override via scan options.
DEFAULT_MAX_BYTES = 256 * 1024
# Small chunks help a small (3B) model. If users plug in a 14B model via
# CODE_SCAN_MODEL they can override these via the start-scan options.
DEFAULT_MAX_CHUNK_LINES = 200
DEFAULT_OVERLAP_LINES = 30
DEFAULT_MAX_FILES = 0   # 0 == unlimited


# ---------------------------------------------------------------------------
# In-process registry of running scans
# ---------------------------------------------------------------------------

class _ScanHandle:
    """In-memory bookkeeping for one running scan (not persisted itself --
    the canonical state lives in the `code_scan` row)."""

    def __init__(self, scan_id: int):
        self.scan_id = scan_id
        self.cancel_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.current_file: Optional[str] = None
        self.current_chunk: int = 0
        self.started_at = time.time()


_active: Dict[int, _ScanHandle] = {}
_active_lock = threading.Lock()


def get_handle(scan_id: int) -> Optional[_ScanHandle]:
    with _active_lock:
        return _active.get(scan_id)


def runtime_status(scan_id: int) -> Dict[str, Any]:
    """In-memory progress info that is NOT persisted on every chunk."""
    h = get_handle(scan_id)
    if h is None:
        return {"active": False}
    return {
        "active": h.thread is not None and h.thread.is_alive(),
        "current_file": h.current_file,
        "current_chunk": h.current_chunk,
        "cancel_requested": h.cancel_event.is_set(),
        "elapsed_s": round(time.time() - h.started_at, 2),
    }


def cancel(scan_id: int) -> bool:
    h = get_handle(scan_id)
    if h is None:
        return False
    h.cancel_event.set()
    return True


# ---------------------------------------------------------------------------
# Walking + chunking
# ---------------------------------------------------------------------------

def _is_probably_text(head: bytes) -> bool:
    """Quick binary detector: lots of NULs or many non-ASCII bytes -> skip."""
    if not head:
        return True
    if b"\x00" in head:
        return False
    n_high = sum(1 for b in head if b > 127)
    # allow some UTF-8 (high bytes), but if >30% are high, treat as binary
    return n_high / len(head) < 0.30


def iter_source_files(
    root: Path, *, extensions: Dict[str, str], skip_dirs: set,
    max_bytes: int, max_files: int = 0,
    restrict_to: Optional[set] = None,
) -> Iterator[Tuple[Path, str]]:
    """Yield (path, language) for each source file under root.

    If `restrict_to` is provided, only yield files whose POSIX-relative
    path (relative to `root`) is in that set. This is how `git_since`
    diff-only scans are implemented: walk the tree normally so we still
    apply the binary/size/extension filters, but skip anything not in the
    git-changed set.
    """
    yielded = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # in-place prune of skip_dirs
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            lang = extensions.get(ext)
            if not lang:
                continue
            p = Path(dirpath) / fn
            if restrict_to is not None:
                rel_posix = p.relative_to(root).as_posix()
                if rel_posix not in restrict_to:
                    continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > max_bytes:
                continue
            yield p, lang
            yielded += 1
            if max_files and yielded >= max_files:
                return


def _git_changed_files(root: Path, since: str) -> Optional[set]:
    """Return the set of files changed since `since` (a git ref or sha).

    The set is POSIX-relative paths from `root`. Returns None if `root`
    is not a git working tree, or if the git invocation fails -- the
    caller should treat that as a hard error (we don't want to silently
    fall back to a full scan when the user asked for diff-only).

    Includes:
      - tracked files modified between `since` and HEAD (A/C/M/R)
      - untracked files (so a brand-new file the user hasn't committed
        yet is also reviewed)
    Excludes deletions (nothing to scan).
    """
    if not (root / ".git").exists():
        # also check parent dirs in case `root` is a subdir of a repo
        try:
            check = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(root), capture_output=True, text=True, timeout=5,
            )
            if check.returncode != 0 or check.stdout.strip() != "true":
                return None
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

    out: set = set()
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", since],
            cwd=str(root), capture_output=True, text=True, timeout=15,
        )
        if diff.returncode != 0:
            return None
        for line in diff.stdout.splitlines():
            line = line.strip()
            if line:
                out.add(line)
        # Also include untracked-but-not-ignored files (new files the
        # developer hasn't `git add`-ed yet are exactly the ones a
        # pre-commit reviewer cares about most).
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=str(root), capture_output=True, text=True, timeout=15,
        )
        if untracked.returncode == 0:
            for line in untracked.stdout.splitlines():
                line = line.strip()
                if line:
                    out.add(line)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return out


def chunk_file(
    text: str, *, max_lines: int, overlap: int,
) -> List[Tuple[int, int, str]]:
    """Split `text` into overlapping line windows.

    Returns a list of (start_line_1indexed, end_line_1indexed_inclusive, chunk_text).
    """
    if not text:
        return []
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []
    if n <= max_lines:
        return [(1, n, text)]
    out: List[Tuple[int, int, str]] = []
    stride = max(1, max_lines - overlap)
    i = 0
    while i < n:
        j = min(i + max_lines, n)
        out.append((i + 1, j, "\n".join(lines[i:j])))
        if j >= n:
            break
        i += stride
    return out


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# The scan loop (runs on a dedicated daemon thread)
# ---------------------------------------------------------------------------

def _run_scan(scan_id: int, root: Path, options: Dict[str, Any]) -> None:
    handle = get_handle(scan_id)
    extensions: Dict[str, str] = options.get("extensions") or DEFAULT_EXTENSIONS
    skip_dirs = set(options.get("skip_dirs") or DEFAULT_SKIP_DIRS)
    max_bytes = int(options.get("max_bytes") or DEFAULT_MAX_BYTES)
    max_chunk_lines = int(options.get("max_chunk_lines") or DEFAULT_MAX_CHUNK_LINES)
    overlap_lines = int(options.get("overlap_lines") or DEFAULT_OVERLAP_LINES)
    max_files = int(options.get("max_files") or DEFAULT_MAX_FILES)
    persist_low = bool(options.get("persist_low", False))
    git_since = options.get("git_since") or None

    try:
        with db.session() as conn:
            db.update_code_scan(conn, scan_id, status="running")

        # Phase 1: enumerate -- cheap, also lets us populate total_files.
        # If `git_since` was set, restrict to files git knows have changed
        # since that ref (plus untracked). If git is unavailable or the
        # path isn't a repo, fail loudly: silently scanning the whole
        # tree when the user asked for a diff would be the wrong default.
        restrict_to: Optional[set] = None
        if git_since:
            restrict_to = _git_changed_files(root, git_since)
            if restrict_to is None:
                with db.session() as conn:
                    db.update_code_scan(
                        conn, scan_id, status="error",
                        error=(
                            f"git_since={git_since!r} but {root} is not a "
                            "git working tree (or git is not on PATH). "
                            "Remove git_since to scan everything."
                        ),
                        finished=True,
                    )
                return
            if not restrict_to:
                # No changed files -- that's not an error, it's a no-op
                # done scan with zero work. Useful for CI: "scan the diff
                # of an empty PR" should succeed instantly.
                with db.session() as conn:
                    db.update_code_scan(
                        conn, scan_id, status="done",
                        total_files=0, scanned_files=0,
                        skipped_files=0, findings_count=0,
                        finished=True,
                    )
                return

        files = list(iter_source_files(
            root, extensions=extensions, skip_dirs=skip_dirs,
            max_bytes=max_bytes, max_files=max_files,
            restrict_to=restrict_to,
        ))
        with db.session() as conn:
            db.update_code_scan(conn, scan_id, total_files=len(files))

        scanned = 0
        skipped = 0
        findings_count = 0

        # Phase 2: per-file decode
        for p, lang in files:
            if handle and handle.cancel_event.is_set():
                with db.session() as conn:
                    db.update_code_scan(
                        conn, scan_id, status="cancelled",
                        scanned_files=scanned, skipped_files=skipped,
                        findings_count=findings_count, finished=True,
                    )
                return

            rel = str(p.relative_to(root))
            if handle:
                handle.current_file = rel
                handle.current_chunk = 0

            try:
                head = p.read_bytes()[:4096]
            except OSError:
                skipped += 1
                continue
            if not _is_probably_text(head):
                skipped += 1
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skipped += 1
                continue

            file_sha = _sha256_text(text)
            chunks = chunk_file(
                text, max_lines=max_chunk_lines, overlap=overlap_lines,
            )

            for chunk_i, (start_line, end_line, chunk_text) in enumerate(chunks):
                if handle and handle.cancel_event.is_set():
                    break
                if handle:
                    handle.current_chunk = chunk_i

                result = nla_client.decode_code(
                    chunk_text, language=lang, path_hint=rel,
                )
                if not result.get("ok"):
                    # Soft-fail per chunk; record nothing, keep scanning.
                    continue
                findings = result.get("findings") or []
                summary = result.get("summary")
                if not findings:
                    continue
                # Persist findings (filter out info-level unless persist_low)
                with db.session() as conn:
                    for f in findings:
                        sev = (f.get("severity") or "info").lower()
                        # Default policy: keep low+. Only `info` is dropped
                        # unless the user explicitly asks for it via
                        # persist_low. A screening tool should surface
                        # 'low' -- the user can sort/filter in the UI.
                        if sev == "info" and not persist_low:
                            continue
                        db.record_code_finding(
                            conn,
                            scan_id=scan_id,
                            file_path=rel,
                            file_sha256=file_sha,
                            chunk_index=chunk_i,
                            chunk_start_line=start_line,
                            chunk_end_line=end_line,
                            language=lang,
                            kind=f.get("kind") or "external_input",
                            severity=sev,
                            line_hint=f.get("line_hint"),
                            excerpt=f.get("excerpt") or "",
                            explanation=f.get("explanation"),
                            chunk_summary=summary,
                        )
                        findings_count += 1

            scanned += 1
            # Persist progress periodically (every file is fine; files are slow)
            with db.session() as conn:
                db.update_code_scan(
                    conn, scan_id,
                    scanned_files=scanned, skipped_files=skipped,
                    findings_count=findings_count,
                )

        with db.session() as conn:
            db.update_code_scan(
                conn, scan_id, status="done",
                scanned_files=scanned, skipped_files=skipped,
                findings_count=findings_count, finished=True,
            )
    except Exception as e:
        with db.session() as conn:
            db.update_code_scan(
                conn, scan_id, status="error",
                error=f"{type(e).__name__}: {e}", finished=True,
            )
    finally:
        with _active_lock:
            _active.pop(scan_id, None)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def start_scan(
    root_path: str, *, label: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate inputs, create the scan row, spawn the worker thread.

    Returns {"scan_id": int} on success, or {"error": str}.
    """
    root = Path(root_path).expanduser().resolve()
    if not root.exists():
        return {"error": f"path does not exist: {root}"}
    if not root.is_dir():
        return {"error": f"path is not a directory: {root}"}

    # quick readiness check -- fail fast if Ollama isn't reachable
    cs = nla_client.code_status()
    if not cs.get("ready"):
        return {
            "error": (
                f"code-scan model {cs.get('model')!r} not reachable. "
                f"Pull it with: ollama pull {cs.get('model')}"
            ),
        }

    opts = dict(options or {})
    with db.session() as conn:
        scan_id = db.create_code_scan(
            conn, root_path=str(root), label=label,
            model=cs.get("model"),
            prompt_version=cs.get("prompt_version"),
            options=opts,
        )

    handle = _ScanHandle(scan_id)
    with _active_lock:
        _active[scan_id] = handle
    t = threading.Thread(
        target=_run_scan, args=(scan_id, root, opts),
        name=f"code-scan-{scan_id}", daemon=True,
    )
    handle.thread = t
    t.start()
    return {"scan_id": scan_id}
