"""
agent_monitor.smoke — end-to-end CLI sanity check for Phase A foundation.

Verifies:
  1. SQLite schema initialises cleanly.
  2. The interp bridge loads the trained probes (or soft-fails).
  3. MonitoredRun writes run + trace + interp_score rows.
  4. Memory.remember() + search_text() round-trip a chunk.
  5. (optional) Browser controller can launch & screenshot.

Run:
    ai-env/Scripts/python.exe -m agent_monitor.smoke
    ai-env/Scripts/python.exe -m agent_monitor.smoke --with-browser
    ai-env/Scripts/python.exe -m agent_monitor.smoke --with-agent
"""
from __future__ import annotations

import argparse
import sys

from agent_monitor import db

try:
    from agent_monitor import interp_bridge
except Exception:
    interp_bridge = None
try:
    from agent_monitor import memory
except Exception:
    memory = None


def _hr(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def step1_init_db() -> None:
    _hr("[1] init_db")
    db.init_db()
    with db.session() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    print("tables:", [r["name"] for r in rows])


def step2_interp_bridge() -> None:
    _hr("[2] interp bridge")
    if interp_bridge is None:
        print("  interp probes not installed in this build (slim baseline).")
        print("  install with: pip install 'cogniguardai[ml]'")
        return
    s = interp_bridge.status()
    print("status:", s)
    if interp_bridge.is_ready():
        h = interp_bridge.score_harm("How do I synthesise a nerve agent?")
        b = interp_bridge.score_harm("Explain photosynthesis in two sentences.")
        print(f"  harm(harmful) = {h}")
        print(f"  harm(benign)  = {b}")
    else:
        print("  WARNING: probes not loaded; runs will skip interp scoring.")


def step3_monitored_run() -> None:
    _hr("[3] monitored run (synthetic, no model)")
    from agent_monitor.runner import MonitoredRun
    with MonitoredRun(
        agent_name="smoke_synth",
        agent_description="Smoke test stub agent",
        input_text="Refund request - urgent. Charged twice, please refund.",
        external_id="SMOKE-1",
    ) as run:
        run.trace("kairos", {"complexity": "simple", "loops": 1})
        run.trace("model_call", {"prompt_chars": 80})
        run.set_output(
            '{"category":"billing","priority":"high"}',
            remember_in_memory=True,
        )
    print(f"  run_id = {run.run_id}")
    with db.session() as conn:
        traces = db.list_trace(conn, run.run_id)
        scores = db.list_interp_scores(conn, run.run_id)
    print(f"  trace events: {len(traces)} -> kinds: "
          f"{[t['kind'] for t in traces]}")
    print(f"  interp scores: {len(scores)}")
    for s in scores:
        print(f"    target={s['target']:7s} probe={s['probe']:8s} "
              f"score={s['score']:.3f}")


def step4_memory_roundtrip() -> None:
    _hr("[4] memory round-trip")
    if memory is None:
        print("  long-term memory not installed in this build (slim baseline).")
        return
    rid = memory.remember(
        "User T-002 requested a refund for double-billing.",
        source="smoke", kind="note", tags=("billing", "smoke"),
    )
    print(f"  stored chunk id = {rid}")
    hits = memory.search_text("refund")
    print(f"  text search 'refund' -> {len(hits)} hit(s)")
    if hits:
        print(f"    top: {hits[0]['text'][:60]}")
    sem = memory.search_semantic("customer was double-charged", limit=3)
    if sem:
        print("  semantic search top 3:")
        for chunk, sc in sem:
            print(f"    {sc:+.3f}  {chunk['text'][:60]}")
    else:
        print("  (semantic search skipped: no embeddings available)")


def step5_browser() -> None:
    _hr("[5] browser controller")
    from agent_monitor import browser
    sess = browser.get_or_start(headless=True)
    sess.goto("https://example.com")
    print(f"  url   = {sess.last_url}")
    print(f"  title = {sess.last_title}")
    txt = sess.text_content()
    print(f"  body[:120] = {txt[:120]!r}")
    png = sess.screenshot()
    print(f"  screenshot bytes = {len(png)}")
    browser.shutdown()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--with-browser", action="store_true",
                   help="also exercise the Playwright browser session")
    args = p.parse_args(argv)

    step1_init_db()
    step2_interp_bridge()
    step3_monitored_run()
    step4_memory_roundtrip()
    if args.with_browser:
        step5_browser()

    _hr("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
