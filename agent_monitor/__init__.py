"""
agent_monitor — self-hosted AI-agents monitoring platform.

Wraps the existing automations (customer_support, sop_processor, meta_harness)
with persistent memory, browser automation, live trace capture, and the
production-grade interp probes (interp/ + interp_real/).

Public modules:
    db.py        SQLite schema + thin CRUD layer (sync + async)
    memory.py    Long-term memory store with optional semantic search
    browser.py   Playwright wrapper (start / navigate / screenshot)
    runner.py    Wraps existing automations, emits trace events, scores
                 with interp probes, persists everything to SQLite.
    interp_bridge.py  Loads the trained probes once, scores text on demand.
    api.py       FastAPI app (Phase B)
    desktop.py   pywebview launcher (Phase E)

Beginner note:
    Nothing in this package mutates user data outside agent_monitor/data/.
    The SQLite database lives at agent_monitor/data/monitor.db.
"""

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent

# When frozen by PyInstaller, the package dir lives inside sys._MEIPASS
# which is a read-only temp extract. Persistent data must live elsewhere:
# %LOCALAPPDATA%/AgentMonitor on Windows. In dev, keep it next to the
# package for easy inspection.
# Order of precedence for the persistent data dir:
#   1. AGENT_MONITOR_DATA_DIR env var (explicit override; great for tests
#      and for pointing dev mode at the .exe's DB to see the same data)
#   2. frozen mode  -> %LOCALAPPDATA%/AgentMonitor
#   3. dev mode     -> agent_monitor/data/
_override = os.environ.get("AGENT_MONITOR_DATA_DIR")
if _override:
    DATA_DIR = Path(_override)
elif getattr(sys, "frozen", False):
    _root = Path(os.environ.get("LOCALAPPDATA")
                 or os.path.expanduser("~/AppData/Local"))
    DATA_DIR = _root / "AgentMonitor"
else:
    DATA_DIR = PACKAGE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "monitor.db"
