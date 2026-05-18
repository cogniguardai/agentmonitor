"""
agent_monitor.classifiers -- defender-side trace classifiers (v1.8).

Each module exports `classify(trace_text)` returning a dict with at
least `{score, kind, signals}`. Classifiers do NOT modify state; they
just read trace text and report patterns.

Currently ships:
    * `offensive_patterns` -- detects agent traces that exhibit
      patterns associated with offensive security / exploit-dev
      workflows. Reads only TEXT content from `trace_event.payload_json`.
      Ships with a small public-source signature library; emits all
      matches so the user can decide.
"""
