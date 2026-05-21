"""
agent_monitor -- a flight recorder for AI agents.

Records every prompt, tool call, file touched, token spent and dollar
burned to a local SQLite database, then surfaces it through a self-
hosted dashboard (FastAPI + a small JS UI). Local-first by design:
your prompts never leave your machine.

Public modules (slim baseline, ships in `pip install cogniguardai`):
    db.py        SQLite schema + thin CRUD layer
    runner.py    `MonitoredRun` context manager -- the public SDK
    api.py       FastAPI dashboard server
    desktop.py   pywebview launcher (the `agentmonitor` CLI entrypoint)
    pricing.py   public LLM cost data
    seed.py      demo data for first-run UX
    adapters/    LLM-specific wrappers (openai, anthropic, langchain, ...)

Optional Phase-2 features (opt-in extras, lazy-loaded by api.py):
    interp_bridge / memory / nla_*       (`pip install 'cogniguardai[ml]'`)
    browser.py                           (`pip install 'cogniguardai[browser]'`)
    code_scan / classifiers              (in source, not yet on PyPI)

Persistent data lives at:
    1. $AGENT_MONITOR_DATA_DIR if set                (test override)
    2. %LOCALAPPDATA%/AgentMonitor when frozen       (.exe install)
    3. agent_monitor/data/                           (dev mode)
"""

__version__ = "0.1.1"

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
