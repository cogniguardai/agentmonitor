"""
agent_monitor.desktop — pywebview launcher.

Boots the FastAPI/uvicorn server in a daemon thread, waits for it to
respond on /api/status, then opens a native Windows window pointing at
the local server. This is what gets bundled by PyInstaller into
AgentMonitor.exe.

Run directly:
    ai-env/Scripts/python.exe -m agent_monitor.desktop
    ai-env/Scripts/python.exe -m agent_monitor.desktop --no-window
    ai-env/Scripts/python.exe -m agent_monitor.desktop --no-window --port 9000

Flags:
    --no-window  Run uvicorn in the foreground only -- no native window.
                 Useful for headless servers and post-build smoke tests.
    --port N     Bind the server on port N instead of the default 8765.
                 If the port is busy, an OS-assigned ephemeral is used.

Notes for beginners:
  - On Windows, pywebview must run on the MAIN thread. So uvicorn lives
    in a background thread and webview.start() blocks the main thread.
  - In --no-window mode there's no second thread; uvicorn runs directly
    on the main thread so Ctrl-C / SIGTERM cleanly stop the process.
  - We try port 8765 first (matches the dev URL); if it's busy we pick
    a free ephemeral port. Useful when an old server is still running.
  - When the window closes, the process exits and the daemon thread
    (uvicorn) dies with it.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.request

import uvicorn

from agent_monitor.api import app as fastapi_app

LOG_PATH = os.path.join(tempfile.gettempdir(), "AgentMonitor.log")


def _log(msg: str) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


def _free_port(preferred: int = 8765) -> int:
    """Return preferred if available, else an OS-assigned ephemeral port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", preferred))
        s.close()
        return preferred
    except OSError:
        s.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _run_server(host: str, port: int) -> None:
    try:
        config = uvicorn.Config(
            fastapi_app,                  # pass the app object directly
            host=host, port=port,
            log_level="warning",
            lifespan="on",
            loop="asyncio",
            # Critical for PyInstaller --noconsole builds: uvicorn's
            # default log config calls sys.stderr.isatty() to decide on
            # ANSI colors, and sys.stderr is None when there is no
            # console subsystem. Disabling the dict-config entirely
            # leaves Python's root logger -- good enough for our use.
            log_config=None,
            access_log=False,
        )
        uvicorn.Server(config).run()
    except Exception:
        _log("=== uvicorn server thread crashed ===")
        _log(traceback.format_exc())


def _ensure_std_streams() -> None:
    """PyInstaller bundles built with `console=False` start with
    sys.stdout / sys.stderr / sys.stdin set to None. Plenty of third-
    party libs (uvicorn, click, urllib3, certifi diagnostics, ...) do
    things like `sys.stderr.write(...)` or `sys.stderr.isatty()` and
    crash on None. We replace them with /dev/null-style writers so all
    such calls no-op safely. This is purely defensive plumbing -- our
    own diagnostics go through `_log()` to the file, not stdio."""
    devnull = open(os.devnull, "w", encoding="utf-8", buffering=1)
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            setattr(sys, name, devnull)


def _wait_for_server(url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="AgentMonitor",
        description="Launch the AgentMonitor server (+ desktop window by default).",
    )
    p.add_argument(
        "--no-window", action="store_true",
        help="Run uvicorn only -- no native window. Useful for headless "
             "servers and post-build smoke tests.",
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="Preferred port to bind (default 8765). If busy, an ephemeral "
             "port is chosen.",
    )
    p.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind. Default 127.0.0.1; use 0.0.0.0 only when you "
             "explicitly want LAN access.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # MUST run before any third-party import that may write to stdio
    # (uvicorn, asyncio, etc.). See _ensure_std_streams docstring.
    _ensure_std_streams()

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    _log(f"--- AgentMonitor boot {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    _log(f"sys.executable = {sys.executable}")
    _log(f"frozen         = {getattr(sys, 'frozen', False)}")
    _log(f"_MEIPASS       = {getattr(sys, '_MEIPASS', None)}")
    _log(f"args           = no_window={args.no_window} port={args.port} host={args.host}")

    host = args.host
    port = _free_port(args.port)
    base = f"http://{host}:{port}"
    _log(f"server target  = {base}")

    if args.no_window:
        # Headless: run uvicorn directly on the main thread. Callers
        # discover the actual bound port via the "server target = ..."
        # line in %TEMP%\AgentMonitor.log (stdout is /dev/null on a
        # console=False PyInstaller build).
        _run_server(host, port)
        return

    # Windowed (default): uvicorn in a daemon thread, webview on main.
    t = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    t.start()

    if not _wait_for_server(f"{base}/api/status"):
        _log(f"server did not respond at {base} within 20s; opening window anyway")

    # Lazy import so --no-window mode works on machines without a
    # display server / webview2 runtime installed.
    import webview  # noqa: WPS433
    webview.create_window(
        title="AgentMonitor",
        url=base,
        width=1280,
        height=820,
        min_size=(900, 600),
        background_color="#0e1116",
        text_select=True,
    )
    webview.start()


if __name__ == "__main__":
    main()
