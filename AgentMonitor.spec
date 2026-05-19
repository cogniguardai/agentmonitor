# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for AgentMonitor.exe (Windows bundled desktop app).

Reproducible build:

    # 1. Create build venv (any Python 3.10/3.11/3.12)
    python -m venv ..\_build_venv
    ..\_build_venv\Scripts\pip install pywebview fastapi 'uvicorn[standard]' \
        pydantic httpx rich PyYAML pyinstaller

    # 2. Build, from the repo root, using THIS spec
    ..\_build_venv\Scripts\pyinstaller AgentMonitor.spec --clean --noconfirm

    # 3. Output lands in dist/AgentMonitor.exe (~20 MB)

The spec produces a single-file --noconsole .exe that:
  - bundles agent_monitor/web/ (index.html + static/) as runtime data
  - declares uvicorn's auto-loaded submodules as hidden imports
    (uvicorn discovers them via importlib at runtime; PyInstaller's
    static analysis misses them otherwise)
  - explicitly lists agent_monitor.* modules so a fresh install with
    no .pyc cache still picks them all up
  - leaves stdout / stderr at None for the bootloader; the app's
    own _ensure_std_streams() in agent_monitor/desktop.py replaces
    them with /dev/null writers so third-party libs don't crash
"""
from pathlib import Path

# Repo root, resolved relative to this spec file's location. SPEC is a
# magic global that PyInstaller injects when it executes the spec.
HERE = Path(SPEC).resolve().parent
PKG_DIR = HERE / "agent_monitor"

assert PKG_DIR.is_dir(), (
    f"agent_monitor/ not found at {PKG_DIR}. "
    "Run pyinstaller from the repo root, with this spec at the root too."
)

a = Analysis(
    [str(PKG_DIR / "desktop.py")],
    pathex=[str(HERE)],
    binaries=[],
    datas=[
        # Bundle web/index.html, web/static/app.js, web/static/style.css.
        # PyInstaller copies the directory recursively; the second arg
        # is the path WITHIN the bundle (matches package layout so
        # `Path(__file__).parent / 'web' / ...` resolves correctly at
        # runtime when sys._MEIPASS is on sys.path).
        (str(PKG_DIR / "web"), "agent_monitor/web"),
    ],
    hiddenimports=[
        # uvicorn auto-discovery -- PyInstaller can't see these via
        # static analysis because uvicorn picks them at runtime via
        # importlib based on the installed extras.
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan.on",
        # agent_monitor public modules (slim baseline only -- optional
        # ML / browser / NLA modules are deliberately NOT listed; if
        # the corresponding deps aren't installed, the modules fail
        # to import and api.py's _try_import returns None, which the
        # endpoints already handle with HTTP 503).
        "agent_monitor.api",
        "agent_monitor.db",
        "agent_monitor.runner",
        "agent_monitor.pricing",
        "agent_monitor.adapters",
        "agent_monitor.adapters.openai",
        "agent_monitor.adapters.anthropic",
        "agent_monitor.adapters.langchain",
        "agent_monitor.adapters.ollama",
        "agent_monitor.adapters.smolagents",
        "agent_monitor.adapters.autogen",
        "agent_monitor.adapters.pipeline",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy ML deps that the slim build doesn't ship. Listed
        # explicitly so PyInstaller doesn't accidentally pull them
        # in via transitive references.
        "torch",
        "transformers",
        "playwright",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AgentMonitor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                # --noconsole: no terminal window pops up
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=str(HERE / "marketing" / "favicon.ico"),  # add when we have one
)
