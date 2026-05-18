"""
agent_monitor.seed — populate the dashboard with realistic data.

Runs a small set of representative customer-support tickets through the
existing automation (Qwen via Ollama + KAIROS) using the MonitoredRun
context manager. Each ticket produces:
    - a row in `run`
    - several rows in `trace_event`
    - 6 rows in `interp_score` (3 probes x input + output)
    - 1 row in `memory_chunk` (the output, embedded)

Run:
    # populate dev DB (agent_monitor/data/monitor.db)
    ai-env\\Scripts\\python.exe -m agent_monitor.seed

    # populate the .exe's DB
    $env:AGENT_MONITOR_DATA_DIR = "$env:LOCALAPPDATA\\AgentMonitor"
    ai-env\\Scripts\\python.exe -m agent_monitor.seed

Cost note: each ticket triggers a real Ollama call (~5-15s on CPU), so
the full set takes ~1-2 minutes.
"""
from __future__ import annotations

import sys

from agent_monitor import DB_PATH, db
from agent_monitor.runner import run_customer_support_tickets

# Mix of categories, priorities, and ambiguity levels.
TICKETS = [
    {
        "id": "T-100",
        "subject": "Unable to log in after password reset",
        "body": "I reset my password 30 minutes ago and now I cannot log in. "
                "The site says 'invalid credentials' even though I'm copying "
                "the new password directly from the email.",
        "priority": "high",
    },
    {
        "id": "T-101",
        "subject": "Charged twice for monthly subscription",
        "body": "My credit card statement shows two identical $29.99 charges "
                "from your service on the same day. Please refund the duplicate "
                "charge as soon as possible.",
        "priority": "high",
    },
    {
        "id": "T-102",
        "subject": "Feature request: dark mode",
        "body": "Would love to see a dark mode option in the web app. The current "
                "white background is hard on my eyes during long sessions.",
        "priority": "low",
    },
    {
        "id": "T-103",
        "subject": "Export to CSV produces empty file",
        "body": "When I click 'Export to CSV' in the reporting section, I get "
                "a 0-byte file. Tried in Chrome and Firefox, same result. "
                "Date range is 2026-04-01 to 2026-04-30, ~5000 rows expected.",
        "priority": "medium",
    },
    {
        "id": "T-104",
        "subject": "Cancelling my account",
        "body": "Service is no longer a fit for our team. Please cancel my "
                "Pro subscription effective end of month and confirm in writing.",
        "priority": "medium",
    },
    {
        "id": "T-105",
        "subject": "API returns 500 on /v1/users/search",
        "body": "Our integration started failing this morning. POST to "
                "/v1/users/search with any payload returns 500 Internal Server "
                "Error. Was working fine yesterday. Production traffic blocked.",
        "priority": "critical",
    },
]


def _hr(t: str) -> None:
    print()
    print("=" * 72)
    print(f"  {t}")
    print("=" * 72)


def main() -> int:
    _hr(f"AgentMonitor seed -> {DB_PATH}")
    db.init_db()

    print(f"\n  ingesting {len(TICKETS)} tickets via Ollama (this takes ~1-2 min)\n")
    run_ids = run_customer_support_tickets(TICKETS)

    _hr("Summary")
    with db.session() as conn:
        for rid in run_ids:
            cur = conn.execute(
                "SELECT external_id, status, elapsed_ms, "
                "(SELECT MAX(score) FROM interp_score "
                " WHERE run_id=run.id AND probe='harm') AS harm_max, "
                "(SELECT COUNT(*) FROM trace_event WHERE run_id=run.id) AS n_trace "
                "FROM run WHERE id = ?", (rid,)
            )
            r = cur.fetchone()
            harm = "—" if r["harm_max"] is None else f"{r['harm_max']:.2f}"
            elapsed = "?" if r["elapsed_ms"] is None else f"{r['elapsed_ms']/1000:.1f}s"
            flag = " ⚑" if r["harm_max"] and r["harm_max"] >= 0.7 else ""
            print(f"  run {rid:>3} {r['external_id']:6s} {r['status']:7s} "
                  f"harm={harm}{flag}  trace={r['n_trace']:>2}  {elapsed}")

    _hr("SEED OK")
    print("  Open the dashboard:")
    print("    ai-env\\Scripts\\python.exe -m agent_monitor.run_server")
    print("    http://127.0.0.1:8765")
    return 0


if __name__ == "__main__":
    sys.exit(main())
