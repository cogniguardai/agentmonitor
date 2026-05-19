"""
agent_monitor.seed -- populate the dashboard with realistic demo data.

This is a self-contained, no-LLM-required seeder for first-run UX.
Drops 6 representative customer-support runs into the SQLite DB so the
dashboard has something to show when the user opens it for the first
time.

Each seeded run produces:
    - one row in `run` (status=done, plausible elapsed_ms)
    - 3-5 rows in `trace_event` (kairos, model_call, model_response)
    - the output text is set to a plausible triage JSON

Compared to the old `seed.py`, this version:
    - does NOT call Ollama or any LLM
    - does NOT depend on Mythos's `core.engine` / `core.kairos`
    - runs in <2 seconds on any machine

Run:
    # populate dev DB (agent_monitor/data/monitor.db)
    python -m agent_monitor.seed

    # populate the .exe's DB (Windows installer)
    $env:AGENT_MONITOR_DATA_DIR = "$env:LOCALAPPDATA\\AgentMonitor"
    python -m agent_monitor.seed
"""
from __future__ import annotations

import sys

from agent_monitor import DB_PATH, db
from agent_monitor.runner import MonitoredRun

# Each tuple is (ticket_dict, kairos_result, plausible_output_json).
# The "output" string is what a real LLM would have produced; we just
# record it directly without making any model call.
DEMO_RUNS = [
    (
        {
            "id": "T-100",
            "subject": "Unable to log in after password reset",
            "body": (
                "I reset my password 30 minutes ago and now I cannot log in. "
                "The site says 'invalid credentials' even though I'm copying "
                "the new password directly from the email."
            ),
            "priority": "high",
        },
        {"complexity": "moderate", "loops": 6},
        '{"category":"auth","priority":"high",'
        '"reason":"password reset flow appears broken; user blocked from access"}',
    ),
    (
        {
            "id": "T-101",
            "subject": "Charged twice for monthly subscription",
            "body": (
                "My credit card statement shows two identical $29.99 charges "
                "from your service on the same day. Please refund the duplicate "
                "charge as soon as possible."
            ),
            "priority": "high",
        },
        {"complexity": "simple", "loops": 4},
        '{"category":"billing","priority":"high",'
        '"reason":"duplicate charge; clear refund-eligible event"}',
    ),
    (
        {
            "id": "T-102",
            "subject": "Feature request: dark mode",
            "body": (
                "Would love to see a dark mode option in the web app. The "
                "current white background is hard on my eyes during long "
                "sessions."
            ),
            "priority": "low",
        },
        {"complexity": "simple", "loops": 2},
        '{"category":"feature","priority":"low",'
        '"reason":"non-blocking enhancement request"}',
    ),
    (
        {
            "id": "T-103",
            "subject": "Export to CSV produces empty file",
            "body": (
                "When I click 'Export to CSV' in the reporting section, I get "
                "a 0-byte file. Tried in Chrome and Firefox, same result. "
                "Date range is 2026-04-01 to 2026-04-30, ~5000 rows expected."
            ),
            "priority": "medium",
        },
        {"complexity": "moderate", "loops": 5},
        '{"category":"technical","priority":"medium",'
        '"reason":"export pipeline regression; reproducible across browsers"}',
    ),
    (
        {
            "id": "T-104",
            "subject": "Cancelling my account",
            "body": (
                "Service is no longer a fit for our team. Please cancel my "
                "Pro subscription effective end of month and confirm in writing."
            ),
            "priority": "medium",
        },
        {"complexity": "simple", "loops": 3},
        '{"category":"retention","priority":"medium",'
        '"reason":"polite cancellation request; standard offboarding"}',
    ),
    (
        {
            "id": "T-105",
            "subject": "API returns 500 on /v1/users/search",
            "body": (
                "Our integration started failing this morning. POST to "
                "/v1/users/search with any payload returns 500 Internal Server "
                "Error. Was working fine yesterday. Production traffic blocked."
            ),
            "priority": "critical",
        },
        {"complexity": "complex", "loops": 8},
        '{"category":"technical","priority":"critical",'
        '"reason":"production API outage affecting customer integration"}',
    ),
]


def _hr(t: str) -> None:
    print()
    print("=" * 72)
    print(f"  {t}")
    print("=" * 72)


def _seed_one(ticket: dict, kairos: dict, output_json: str) -> int:
    """Record a single demo run via the public MonitoredRun SDK."""
    text = f"{ticket.get('subject','')}. {ticket.get('body','')}"
    with MonitoredRun(
        agent_name="customer_support_demo",
        agent_description="Demo support-triage agent (seeded data, no LLM call)",
        input_text=text,
        external_id=ticket.get("id"),
        meta={
            "subject": ticket.get("subject"),
            "priority_in": ticket.get("priority"),
            "complexity": kairos["complexity"],
            "loops": kairos["loops"],
        },
        score_input=False,           # interp probes off in seed
    ) as run:
        run.trace("kairos", kairos)
        run.trace("model_call", {
            "prompt_chars": len(text),
            "loops": kairos["loops"],
            "model": "demo-static-output",
        })
        run.trace("model_response", {
            "raw_chars": len(output_json),
            "parsed_ok": True,
        })
        run.set_output(
            output_json,
            score=False,             # no interp probes in seed
            remember_in_memory=False,
        )
    return run.run_id


def main() -> int:
    _hr(f"AgentMonitor seed -> {DB_PATH}")
    db.init_db()

    print(f"\n  ingesting {len(DEMO_RUNS)} demo tickets (no LLM call, ~1 sec)\n")
    run_ids = [_seed_one(t, k, o) for (t, k, o) in DEMO_RUNS]

    _hr("Summary")
    with db.session() as conn:
        for rid in run_ids:
            r = conn.execute(
                "SELECT external_id, status, elapsed_ms, "
                "(SELECT COUNT(*) FROM trace_event WHERE run_id=run.id) AS n_trace "
                "FROM run WHERE id = ?", (rid,),
            ).fetchone()
            elapsed = "?" if r["elapsed_ms"] is None else f"{r['elapsed_ms']:>4}ms"
            print(f"  run {rid:>3} {(r['external_id'] or '-'):8s} "
                  f"{r['status']:7s} trace={r['n_trace']:>2}  {elapsed}")

    _hr("SEED OK")
    print("  Open the dashboard:")
    print("    agentmonitor")
    print("    # or, headless:")
    print("    python -m agent_monitor.run_server")
    print("    # then browse http://127.0.0.1:8765")
    return 0


if __name__ == "__main__":
    sys.exit(main())
