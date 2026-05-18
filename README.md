# AgentMonitor

> *See what your AI agent actually did.*

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

> **Status:** the PyPI namespace [`cogniguardai`](https://pypi.org/project/cogniguardai/)
> is reserved (v0.0.1 placeholder). The first functional release lands as
> **v0.1.0**. Until then, the only working install is from source &mdash;
> see _Run from source_ below.

When v0.1.0 ships:

```bash
pip install cogniguardai
agentmonitor   # opens the dashboard at http://localhost:8765
```

### Run from source (works today)

```bash
git clone https://github.com/cogniguardai/agentmonitor.git
cd agentmonitor
pip install -r requirements.txt
python -m agent_monitor.run_server
# then open http://localhost:8765
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