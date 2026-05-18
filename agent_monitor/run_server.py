"""
agent_monitor.run_server — uvicorn launcher.

Usage:
    ai-env/Scripts/python.exe -m agent_monitor.run_server
    ai-env/Scripts/python.exe -m agent_monitor.run_server --reload --port 8765
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    import uvicorn
    uvicorn.run(
        "agent_monitor.api:app",
        host=args.host, port=args.port, reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
