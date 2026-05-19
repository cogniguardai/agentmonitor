# AgentMonitor

[![Live](https://img.shields.io/badge/Live-cogniguardai.com-0e8a16?style=flat-square&logo=cloudflare&logoColor=white)](https://cogniguardai.com/) [![PyPI](https://img.shields.io/pypi/v/cogniguardai.svg?style=flat-square&label=PyPI&color=8b5cf6)](https://pypi.org/project/cogniguardai/) [![Status](https://img.shields.io/badge/Status-live-0e8a16?style=flat-square)](https://github.com/cogniguardai/agentmonitor#install) [![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

> *See what your AI agent actually did.*

**Live at <https://cogniguardai.com/>** &mdash; **v0.1.0 is on PyPI:** `pip install cogniguardai`.

**AgentMonitor** is a flight recorder for AI agents. It records every
prompt, tool call, file touched and dollar spent &mdash; so when your agent
does something weird, you can rewind and see exactly what happened.

- **Local-first.** Runs as a desktop app on your laptop. Your prompts
  never leave your machine.
- **Free.** No signup, no telemetry, no cloud account. Forever.
- **Works with what you use.** Cursor, Claude Code, OpenAI, Anthropic,
  LangChain, AutoGen, Smolagents, Ollama &mdash; and any custom agent.

Made by [CogniGuard AI](https://cogniguardai.com/).

## Install

```bash
pip install cogniguardai
agentmonitor   # opens the dashboard at http://127.0.0.1:8765
```

The slim baseline is ~174&nbsp;KB on the wire and pulls only fastapi,
uvicorn, pydantic, httpx, rich, PyYAML and pywebview &mdash; no torch,
no transformers, no Playwright. Optional features ship as extras:

```bash
pip install 'cogniguardai[ml]'        # interp probes, NLA decoders, Llama Guard
pip install 'cogniguardai[browser]'   # Playwright session controller
pip install 'cogniguardai[all]'       # everything
```

### Run from source

```bash
git clone https://github.com/cogniguardai/agentmonitor.git
cd agentmonitor
pip install -r requirements.txt
python -m agent_monitor.run_server
# then open http://127.0.0.1:8765
```

## What it does

1. **Record.** Every agent run is captured automatically &mdash; prompts,
   tool calls, files touched, tokens spent, money burned.
2. **Replay.** Open any past run and scroll through it like a video.
3. **Rewind.** When something goes wrong, you have the receipts.

## Documentation

- **Marketing site & live demo**: <https://cogniguardai.com/>
- **Issues / feature requests**: [GitHub Issues](https://github.com/cogniguardai/agentmonitor/issues)
- **Email**: <hello@cogniguardai.com>

## Repository layout

```
agent_monitor/   Python package (the recorder + dashboard backend)
marketing/       Source for the marketing site at cogniguardai.com
requirements.txt Python dependencies
```

## License

MIT &mdash; see [`LICENSE`](LICENSE).

## Privacy

AgentMonitor does not phone home. Ever. All data lives in a local
SQLite database on your machine. There is no analytics, no telemetry,
no signup, no account.