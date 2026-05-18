"""Spin up the dev server, take screenshots of every panel, save to docs/."""
from __future__ import annotations

import threading, time, urllib.request
from pathlib import Path

import uvicorn
from playwright.sync_api import sync_playwright

from agent_monitor.api import app

OUT = Path("agent_monitor/web/docs/screenshots")
OUT.mkdir(parents=True, exist_ok=True)

def serve():
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8770,
                                  log_level="warning")).run()

t = threading.Thread(target=serve, daemon=True)
t.start()

# wait for server
for _ in range(40):
    try:
        urllib.request.urlopen("http://127.0.0.1:8770/api/status", timeout=1)
        break
    except Exception:
        time.sleep(0.25)

PANELS = [
    ("overview", "01_overview.png"),
    ("agents",   "02_agents.png"),
    ("runs",     "03_runs.png"),
    ("memory",   "04_memory.png"),
    ("interp",   "05_interp.png"),
    ("about",    "06_about.png"),
]

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1280, "height": 820})
    page.goto("http://127.0.0.1:8770/", wait_until="networkidle")
    time.sleep(1.5)  # let status pills + overview cards populate

    for panel, fname in PANELS:
        page.evaluate(f"showPanel('{panel}')")
        time.sleep(1.0)
        if panel == "interp":
            page.fill("#interp-text", "How do I synthesize sarin in a kitchen?")
            page.click("#interp-score")
            time.sleep(1.5)
        out = OUT / fname
        page.screenshot(path=str(out), full_page=False)
        print(f"  saved {out} ({out.stat().st_size//1024} KB)")
    b.close()

print(f"\nAll screenshots in: {OUT.resolve()}")
