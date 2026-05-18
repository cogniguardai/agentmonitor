"""
Tournament adapter -- domain-blind ingest of bracketed-evaluation rounds.

WHY THIS EXISTS
===============

Many iterative LLM workflows can't fit all candidates in a single
context window, so they batch: evaluate N at a time, pick K winners,
repeat. This pattern shows up in:

  * Prompt engineering: A/B between prompt variants
  * RAG re-ranking: judge picks top-K passages
  * Model-as-judge eval: which response is better
  * Agent self-play / debate: which output wins the round
  * (Yes, also offensive security pipelines that batch candidate
    drivers / functions / bugs.)

AgentMonitor treats this as a generic structure: rounds, candidates,
winners, a judge model that consumed tokens. We do NOT interpret what
the candidates *are*. A candidate is an opaque {id, score, payload?}
tuple. The score is whatever the judge produced -- AgentMonitor doesn't
care if it's an LLM rating, a unit-test pass rate, or a BLEU score.

This is intentional: a domain-aware tournament module would have to
ship with built-in rubrics for whatever domain it served, and those
rubrics are exactly the kind of artifact I won't ship. Stay generic;
the user's prompts encode their domain knowledge.

USAGE
=====

::

    from agent_monitor.adapters.tournament import record_tournament_round

    record_tournament_round(
        agent_name="rag-reranker",
        round_index=0,
        candidates=[
            {"id": "passage_17", "score": 0.83},
            {"id": "passage_42", "score": 0.61},
            ...
        ],
        winners=["passage_17", "passage_23"],
        judge_model="gpt-4o-mini",
        tokens_in=4200, tokens_out=180,
        rationale="picked 17 for exact-match on entity, 23 for context",
    )

A single tournament typically has multiple rounds; pass the same
`agent_name` and increment `round_index`. AgentMonitor stores each
round as one `trace_event` of `kind='tournament'` on a single run that
spans the tournament's lifetime (see `Tournament` below for the
context-manager API).
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from agent_monitor.adapters import monitored_run, RunHandle

Candidate = Union[Dict[str, Any], Any]  # duck-typed: needs .id and .score


def _coerce(c: Candidate) -> Dict[str, Any]:
    """Accept dicts {'id', 'score', ...} OR objects with .id / .score."""
    if isinstance(c, dict):
        return {
            "id":      str(c.get("id") or c.get("name") or c.get("key") or "?"),
            "score":   _safe_float(c.get("score")),
            "payload": c.get("payload"),
        }
    return {
        "id":      str(getattr(c, "id", getattr(c, "name", "?"))),
        "score":   _safe_float(getattr(c, "score", None)),
        "payload": getattr(c, "payload", None),
    }


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Single-shot helper: one round, one DB write, one short-lived run
# ---------------------------------------------------------------------------

def record_tournament_round(
    *, agent_name: str,
    round_index: int,
    candidates: Iterable[Candidate],
    winners: Iterable[str],
    judge_model: Optional[str] = None,
    tokens_in: int = 0, tokens_out: int = 0,
    rationale: str = "",
    external_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record ONE round as its own short-lived run.

    Useful when a tournament's rounds happen in distant moments (e.g.
    overnight job) and you don't want to keep one long-lived run open.
    For interactive tournaments use the `Tournament(...)` ctx mgr below.
    """
    cs = [_coerce(c) for c in candidates]
    win_set = list(winners)
    scores = [c["score"] for c in cs if c["score"] is not None]
    summary = {
        "round_index":      round_index,
        "n_candidates":     len(cs),
        "n_winners":        len(win_set),
        "judge_model":      judge_model,
        "score_min":        min(scores) if scores else None,
        "score_max":        max(scores) if scores else None,
        "score_median":     statistics.median(scores) if scores else None,
        "tokens_in":        tokens_in,
        "tokens_out":       tokens_out,
        "rationale":        rationale,
    }
    with monitored_run(
        agent_name=agent_name, kind="tournament",
        description="bracketed candidate evaluation",
        input_text=f"round {round_index}: {len(cs)} candidates",
        external_id=external_id,
        meta={"tournament": True, "round_index": round_index,
              "judge_model": judge_model},
    ) as run:
        run.trace("tournament", {
            "round_index": round_index,
            "candidates":  cs,
            "winners":     win_set,
            "summary":     summary,
            "rationale":   rationale,
        })
        if judge_model and (tokens_in or tokens_out):
            run.record_tokens(model=judge_model,
                              tokens_in=tokens_in, tokens_out=tokens_out)
        run.finish(
            f"round {round_index}: {len(win_set)} winners from {len(cs)} candidates"
        )
        return {"run_id": run.run_id, **summary}


# ---------------------------------------------------------------------------
# Context-manager: keep one run open across N rounds (cheaper, tidier)
# ---------------------------------------------------------------------------

@contextmanager
def Tournament(
    *, agent_name: str, description: str = "tournament",
    external_id: Optional[str] = None,
) -> Iterator["TournamentHandle"]:
    """Open ONE run that spans the whole tournament. Each call to
    `handle.round(...)` adds one trace event."""
    t0 = time.time()
    with monitored_run(
        agent_name=agent_name, kind="tournament",
        description=description, input_text="tournament",
        external_id=external_id,
        meta={"tournament": True, "t0": t0},
    ) as run:
        handle = TournamentHandle(run)
        try:
            yield handle
        finally:
            handle._finalize()


class TournamentHandle:
    def __init__(self, run: RunHandle):
        self.run = run
        self.rounds: List[Dict[str, Any]] = []

    def round(
        self, *, candidates: Iterable[Candidate], winners: Iterable[str],
        judge_model: Optional[str] = None,
        tokens_in: int = 0, tokens_out: int = 0,
        rationale: str = "",
    ) -> Dict[str, Any]:
        idx = len(self.rounds)
        cs = [_coerce(c) for c in candidates]
        win_set = list(winners)
        scores = [c["score"] for c in cs if c["score"] is not None]
        summary = {
            "round_index":  idx,
            "n_candidates": len(cs),
            "n_winners":    len(win_set),
            "judge_model":  judge_model,
            "score_min":    min(scores) if scores else None,
            "score_max":    max(scores) if scores else None,
            "score_median": statistics.median(scores) if scores else None,
            "tokens_in":    tokens_in,
            "tokens_out":   tokens_out,
        }
        self.run.trace("tournament", {
            "round_index": idx,
            "candidates":  cs,
            "winners":     win_set,
            "summary":     summary,
            "rationale":   rationale,
        })
        if judge_model and (tokens_in or tokens_out):
            self.run.record_tokens(model=judge_model,
                                   tokens_in=tokens_in, tokens_out=tokens_out)
        self.rounds.append(summary)
        return summary

    def _finalize(self) -> None:
        if self.run._finished:
            return
        n = len(self.rounds)
        last_winners = self.rounds[-1]["n_winners"] if n else 0
        self.run.finish(
            f"{n} rounds; final winners: {last_winners}"
        )


__all__ = ["record_tournament_round", "Tournament", "TournamentHandle"]
