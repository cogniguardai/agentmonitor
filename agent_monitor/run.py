"""
agent_monitor.run — single CLI to invoke any wired-in automation.

Usage:
    python -m agent_monitor.run support           # process sample tickets
    python -m agent_monitor.run blog "Topic text" # generate one blog outline
    python -m agent_monitor.run content "Topic"   # 6-stage pipeline
    python -m agent_monitor.run leads             # follow-ups for sample leads
    python -m agent_monitor.run sops              # score sample SOPs
    python -m agent_monitor.run all               # run EVERY automation once

Every item gets a run_id in the dashboard at http://127.0.0.1:8765/runs/<id>.
"""
from __future__ import annotations

import sys
from typing import List

from agent_monitor import db, DB_PATH
from agent_monitor.runner import run_customer_support_tickets
from agent_monitor.wrappers import (
    run_blog_generator_monitored,
    run_content_pipeline_monitored,
    run_lead_followup_monitored,
    run_sop_processor_monitored,
)

SAMPLE_TICKETS = [
    {"id": "T-200", "subject": "Can't log in",
     "body": "Password reset email never arrives.",
     "priority": "high"},
    {"id": "T-201", "subject": "Refund request",
     "body": "Double-charged for monthly subscription.",
     "priority": "high"},
    {"id": "T-202", "subject": "Feature idea: CSV export in dark mode",
     "body": "Would be nice to have the option.",
     "priority": "low"},
]


def _banner(label: str) -> None:
    print()
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)


def _summary(run_ids: List[int]) -> None:
    with db.session() as conn:
        for rid in run_ids:
            r = conn.execute(
                "SELECT r.external_id, r.status, r.elapsed_ms, "
                "(SELECT MAX(score) FROM interp_score "
                " WHERE run_id=r.id AND probe='harm') AS harm "
                "FROM run r WHERE r.id = ?", (rid,)
            ).fetchone()
            if not r:
                continue
            harm = "----" if r["harm"] is None else f"{r['harm']:.2f}"
            elapsed = (r["elapsed_ms"] or 0) / 1000
            print(f"  run {rid:>3}  {(r['external_id'] or '-'):22s} "
                  f"{r['status']:7s} harm={harm}  {elapsed:>6.1f}s")


def _do(cmd: str, arg: str) -> List[int]:
    if cmd == "support":
        return run_customer_support_tickets(SAMPLE_TICKETS)
    if cmd == "blog":
        topic = arg or "How AI changes customer support in 2026"
        return [run_blog_generator_monitored(topic)]
    if cmd == "content":
        topic = arg or "Practical observability for AI agents"
        return run_content_pipeline_monitored(topic)
    if cmd == "leads":
        return run_lead_followup_monitored()
    if cmd == "sops":
        return run_sop_processor_monitored()
    if cmd == "all":
        ids: List[int] = []
        for c in ("support", "blog", "content", "leads", "sops"):
            _banner(f"running: {c}")
            ids += _do(c, "")
        return ids
    raise SystemExit(f"unknown command: {cmd}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1].lower()
    arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    _banner(f"agent_monitor.run  cmd={cmd}  db={DB_PATH}")
    db.init_db()
    run_ids = _do(cmd, arg)

    _banner(f"done - {len(run_ids)} run(s)")
    _summary(run_ids)
    print("\n  open the dashboard:")
    print("    python -m agent_monitor.run_server   # http://127.0.0.1:8765")
    return 0


if __name__ == "__main__":
    sys.exit(main())
