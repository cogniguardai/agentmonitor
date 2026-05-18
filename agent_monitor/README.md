# AgentMonitor

Self-hosted, downloadable observability platform for **any LLM agent**,
with persistent memory, browser automation, and the production interp
probes wired in for governance signals (harm / refusal / hedging).

It started Qwen-only and now ships adapters for any runtime that
produces a chat-style trajectory. Every run carries an `agent_kind`
column and the UI degrades honestly when a runtime can't expose its
activations.

UI vocabulary borrows from
[paperclip](https://github.com/paperclipai/paperclip) ("zero-human
companies"). AgentMonitor is a focused subset — observability + interp —
not a paperclip replacement.

## Universal agent monitoring

| Runtime       | `agent_kind`   | Trace mode | Interp |
|---------------|----------------|------------|--------|
| Qwen via vLLM | `qwen-vllm`    | real-time  | full (residual stream + text-level) |
| Ollama        | `ollama`       | real-time  | text-level only |
| OpenAI        | `openai`       | real-time  | text-level only |
| Anthropic     | `anthropic`    | real-time  | text-level only |
| LangChain     | `langchain`    | real-time (callback) | text-level only |
| AutoGen       | `autogen`      | post-hoc ingest | text-level only |
| smolagents    | `smolagents`   | post-hoc ingest | text-level only |

**Why interp is split that way:** residual-stream probes need our own
model weights and hooks. For hosted APIs (OpenAI/Anthropic) and
arms-length local runtimes (Ollama) we don't have the activations, so
the UI keeps the **text-level** Llama Guard 3 + embedding probes (which
work on any trace text) and explicitly leaves the mechanistic panels
empty. The Interp tab and Live tab both say so out loud.

### Real-time adapters (Ollama / OpenAI / Anthropic)

```python
from agent_monitor.adapters.openai import OpenAIAdapter
adapter = OpenAIAdapter(agent_name="support-bot", model="gpt-4o-mini")
result = adapter.chat([{"role": "user", "content": "Reset my password"}])
# A new row appears in /runs with agent_kind='openai', full trace,
# elapsed_ms, status, and the response text persisted.
```

### LangChain callback

```python
from langchain_openai import ChatOpenAI
from agent_monitor.adapters.langchain import make_agent_monitor_callback

cb = make_agent_monitor_callback(agent_name="rag-chain")
llm = ChatOpenAI(callbacks=[cb])
llm.invoke("…")  # AgentMonitor sees every LLM call + tool call
```

### AutoGen / smolagents (post-hoc ingest)

Their in-process hook APIs change too often to depend on. We ingest
after the run finishes:

```python
# AutoGen
from agent_monitor.adapters.autogen import record_conversation
record_conversation(agent_name="team", messages=groupchat.messages,
                    model_hint="gpt-4o", elapsed_ms=elapsed)

# smolagents
from agent_monitor.adapters.smolagents import record_agent_run
answer = agent.run(prompt)
record_agent_run(agent_name="math-agent", agent=agent,
                 input_text=prompt, output_text=str(answer))
```

Pass `elapsed_ms=` if you want real wall time; otherwise post-hoc ingest
will show near-zero elapsed (which would be a lie).

## What's in the box

```
agent_monitor/
  __init__.py        # package + data dir resolution (dev vs frozen)
  db.py              # SQLite schema (10 tables: agent, run, trace_event,
                     #   interp_score, memory_chunk, nla_decoding, nla_cache,
                     #   code_scan, code_finding, classifier_signal) + sync CRUD
  pricing.py         # v1.7: public LLM list-price table + compute_cost()
  classifiers/       # v1.8: defender-side trace classifiers
    offensive_patterns.py  # 32 public-source signatures (MITRE / LOLDrivers /
                           #   Microsoft WDK / common defender literature)
  adapters/          # universal AgentAdapter protocol +
                     #   ollama / openai / anthropic / langchain /
                     #   autogen / smolagents implementations,
                     # plus v1.7 generic ingest adapters:
                     #   tournament.py  (bracket / candidate-eval rounds)
                     #   pipeline.py    (bespoke pipelines, sensitive flag)
                     #   findings.py    (Semgrep/CodeQL/Bandit ingest)
                     #   sandbox.py     (record_sandbox_run + Sandbox ctx mgr)
  interp_bridge.py   # singleton loader for harm/refusal/hedging probes
  memory.py          # persistent text + semantic-search memory
  browser.py         # Playwright Chromium controller
  runner.py          # MonitoredRun context + customer_support wrapper
  api.py             # FastAPI: REST + WebSocket
  run_server.py      # uvicorn launcher (dev)
  desktop.py         # pywebview native-window launcher (frozen .exe entry)
  smoke.py           # end-to-end CLI sanity check
  web/               # static UI (HTML / CSS / JS)
  data/              # dev-mode SQLite db (frozen mode uses %LOCALAPPDATA%)
```

## Three ways to run

### 1. Dev mode (web in your browser)

```powershell
ai-env\Scripts\python.exe -m agent_monitor.run_server
# open http://127.0.0.1:8765
```

### 2. Dev mode (native window via pywebview)

```powershell
ai-env\Scripts\python.exe -m agent_monitor.desktop
```

### 3. Production .exe (no Python required)

```powershell
ai-env\Scripts\pyinstaller.exe agent_monitor.spec --clean --noconfirm
.\dist\AgentMonitor\AgentMonitor.exe
```

Distribute by zipping the entire `dist\AgentMonitor\` folder:

```powershell
Compress-Archive -Path dist\AgentMonitor\* -DestinationPath AgentMonitor-win64.zip
```

## Build profile

| | |
|---|---|
| Bundle size | ~168 MB unzipped (~70-90 MB zipped) |
| File count | ~644 |
| Cold-start RAM | ~140 MB |
| Excluded heavy deps | `torch`, `transformers`, `tokenizers`, `safetensors`, `accelerate`, `huggingface_hub`, `scipy`, `pandas`, `matplotlib` |

The exclusion list is the big size win: those libs together would push
the bundle to ~1.5 GB. AgentMonitor doesn't need them because it never
runs Qwen / GPT-2 itself — it monitors agents that run elsewhere
(via Ollama for Qwen) and uses the lightweight numpy probes from
`interp/` for governance scoring.

## Persistent data layout

| Mode | Database | Probes |
|---|---|---|
| Dev | `agent_monitor/data/monitor.db` | `interp/artifacts/probe_*.json` |
| Frozen | `%LOCALAPPDATA%\AgentMonitor\monitor.db` | bundled inside `_internal/interp/artifacts/` |

This split is intentional: in dev you want the DB next to the source for
easy inspection; in production the install dir is read-only (under
`Program Files` typically) so persistent data must go to `%LOCALAPPDATA%`.

## Honesty notes

- **Probes are embedding-space.** They run on `nomic-embed-text`
  embeddings of agent inputs/outputs, not on Qwen's residual stream.
  Useful as behavioural signal; not mechanistic proof.
- **Tiny training sets.** Harm / refusal / hedging probes were trained on
  16-24 examples each. Treat as sanity instruments, not safety claims.
- **Mechanistic interp lives in `interp_real/`,** which hooks GPT-2's
  actual residual stream. Excluded from this .exe (would re-add ~1.5 GB).
  Run it from CLI when you need it.
- **Browser automation needs Chromium installed.** Playwright's Python
  package is bundled, but the actual Chromium binary lives in
  `%LOCALAPPDATA%\ms-playwright`. On a fresh machine, run once:
  `playwright install chromium`.

## Troubleshooting

- **App opens but window is blank / "can't connect":** Ollama may be
  down. Open the DevTools (right-click → Inspect) or check
  `%TEMP%\AgentMonitor.log` for the boot trace.
- **All probes show "off":** `interp/artifacts/probe_*.json` weren't
  bundled. Verify with: `dir dist\AgentMonitor\_internal\interp\artifacts`.
- **Browser panel errors:** run `playwright install chromium` once.
- **Windows SmartScreen warns on first launch:** expected for an
  unsigned binary. Click "More info" → "Run anyway". Code-signing the
  .exe is on the future-work list.
