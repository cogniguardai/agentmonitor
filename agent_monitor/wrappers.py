"""
agent_monitor.wrappers — monitored versions of every automation.

Each function here takes the same inputs as the matching script in
`automations/`, runs the same engine + prompts, but records everything
into the AgentMonitor SQLite DB via `MonitoredRun`. The originals in
`automations/` are left untouched.

Pattern per wrapper:
    1. iterate items (tickets, leads, SOPs, blog briefs, pipeline stages)
    2. open a MonitoredRun per item
    3. emit KAIROS + model_call + model_response trace events
    4. set_output(raw text) -> auto-scores + writes to memory

All wrappers return a list of run_ids so the caller can link into the
dashboard at /runs/<id>.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_monitor.runner import MonitoredRun

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


# --- lazy-loaded engine (so importing this module stays cheap) --------------

_engine = None
_kairos = None


def _get_engine():
    global _engine
    if _engine is None:
        from core.engine import RecurrentDepthEngine
        _engine = RecurrentDepthEngine(prefer="auto", verbose=False)
    return _engine


def _get_kairos():
    global _kairos
    if _kairos is None:
        from core.kairos import KairosController
        _kairos = KairosController(BASE / "configs" / "adaptive_compute.yaml")
    return _kairos


def _run_engine(run: MonitoredRun, prompt: str, *, loops: int) -> str:
    """Shared: trace model_call, invoke, trace model_response, return raw."""
    eng = _get_engine()
    run.trace("model_call", {"prompt_chars": len(prompt), "loops": loops})
    if eng.backend != "ollama":
        raw = "(no model: backend != ollama)"
        run.trace("model_response", {"raw_chars": 0, "reason": "no-backend"})
        return raw
    result = eng.reason(prompt, num_loops=loops)
    raw = result.get("final_text", "") or ""
    run.trace("model_response", {
        "raw_chars": len(raw),
        "elapsed_s": result.get("elapsed_seconds"),
    })
    return raw


# ---------------------------------------------------------------------------
# 1. Blog generator  (was automations/blog_generator.py)
# ---------------------------------------------------------------------------

def run_blog_generator_monitored(
    topic: str,
    audience: str = "small business owners",
    word_count: int = 1500,
    loops: int = 15,
) -> int:
    """Generate a blog outline, fully monitored. Returns run_id."""
    input_text = (
        f"Topic: {topic}\nAudience: {audience}\nWord count: {word_count}"
    )
    with MonitoredRun(
        agent_name="blog_generator",
        agent_description="Outline a blog post for a target audience",
        input_text=input_text,
        external_id=None,
        meta={"topic": topic, "audience": audience,
              "word_count": word_count, "loops": loops},
    ) as run:
        prompt = (
            f"Outline a blog post on '{topic}' for {audience} "
            f"of {word_count} words. Return sections + key bullets."
        )
        raw = _run_engine(run, prompt, loops=loops)
        run.set_output(raw)
    return run.run_id


# ---------------------------------------------------------------------------
# 2. Content pipeline  (was automations/content_pipeline.py)
#    Multi-stage pipeline -> one MonitoredRun PER STAGE. A parent run can be
#    added later; for now each stage stands alone so the dashboard shows
#    each step as a first-class row.
# ---------------------------------------------------------------------------

CONTENT_STAGES = [
    ("ideation", 5,  "Brainstorm 5 angles for a piece about: {topic} (audience: {audience})."),
    ("outline",  8,  "Given these angles, write an outline (H2 + key bullets) for: {topic}."),
    ("draft",   15,  "Draft the introduction (~200 words) for: {topic}. Audience: {audience}."),
    ("edit",    10,  "Edit for clarity and remove filler. Input: {topic} piece draft."),
    ("seo",      5,  "List 8 long-tail keywords for: {topic}. Include search intent."),
    ("review",   3,  "Final review checklist for a blog on: {topic}. Flag weak claims."),
]

def run_content_pipeline_monitored(
    topic: str, audience: str = "small business owners",
) -> List[int]:
    """Run the 6-stage pipeline. Returns run_ids per stage."""
    run_ids: List[int] = []
    external_root = f"pipe-{abs(hash(topic)) % 100000}"
    for stage, loops, tpl in CONTENT_STAGES:
        prompt = tpl.format(topic=topic, audience=audience)
        with MonitoredRun(
            agent_name="content_pipeline",
            agent_description="Ideation -> outline -> draft -> edit -> SEO -> review",
            input_text=prompt,
            external_id=f"{external_root}-{stage}",
            meta={"stage": stage, "loops": loops, "topic": topic},
        ) as run:
            run.trace("stage_enter", {"stage": stage, "loops": loops})
            raw = _run_engine(run, prompt, loops=loops)
            run.set_output(raw)
        run_ids.append(run.run_id)
    return run_ids


# ---------------------------------------------------------------------------
# 3. Lead follow-up  (was automations/lead_followup.py)
# ---------------------------------------------------------------------------

SAMPLE_LEADS = [
    {"name": "Alex Thompson", "company": "TechStartup Inc.", "stage": "cold",
     "interest": "AI customer support"},
    {"name": "Maria Garcia",  "company": "Garcia Marketing", "stage": "warm",
     "interest": "Content pipeline"},
    {"name": "David Chen",    "company": "Chen Consulting",  "stage": "hot",
     "interest": "SOP automation"},
]

def run_lead_followup_monitored(
    leads: Optional[List[Dict[str, Any]]] = None, loops: int = 8,
) -> List[int]:
    leads = leads or SAMPLE_LEADS
    run_ids: List[int] = []
    for lead in leads:
        input_text = (
            f"Lead: {lead['name']} at {lead['company']} | "
            f"stage={lead['stage']} | interest={lead['interest']}"
        )
        with MonitoredRun(
            agent_name="lead_followup",
            agent_description="Write a staged follow-up sequence per lead",
            input_text=input_text,
            external_id=f"lead-{lead['name'].split()[0].lower()}",
            meta={"stage": lead["stage"], "interest": lead["interest"]},
        ) as run:
            prompt = (
                f"Write a 3-touch follow-up sequence for {lead['name']} "
                f"at {lead['company']}. Their interest: '{lead['interest']}'. "
                f"Stage: {lead['stage']}. Tone: helpful, not pushy. "
                f"Return as a numbered list with subject + 2-sentence body each."
            )
            raw = _run_engine(run, prompt, loops=loops)
            run.set_output(raw)
        run_ids.append(run.run_id)
    return run_ids


# ---------------------------------------------------------------------------
# 4. SOP processor  (was automations/sop_processor.py)
# ---------------------------------------------------------------------------

SAMPLE_SOPS = [
    {"name": "New Customer Onboarding", "frequency": "per_customer",
     "steps": [
         "Receive new customer signup notification",
         "Send welcome email within 1 hour",
         "Create customer profile in CRM",
         "Schedule kickoff call within 48 hours",
         "Send training materials",
     ]},
    {"name": "Weekly Content Publishing", "frequency": "weekly",
     "steps": [
         "Review content calendar",
         "Draft blog post",
         "Run SEO pass",
         "Schedule social posts",
         "Publish to CMS",
     ]},
]

def run_sop_processor_monitored(
    sops: Optional[List[Dict[str, Any]]] = None, loops: int = 8,
) -> List[int]:
    sops = sops or SAMPLE_SOPS
    run_ids: List[int] = []
    for sop in sops:
        input_text = (
            f"SOP: {sop['name']} (frequency={sop['frequency']})\n"
            + "\n".join(f"- {s}" for s in sop["steps"])
        )
        with MonitoredRun(
            agent_name="sop_processor",
            agent_description="Score each SOP step for automatability",
            input_text=input_text,
            external_id=f"sop-{abs(hash(sop['name'])) % 10000}",
            meta={"frequency": sop["frequency"], "n_steps": len(sop["steps"])},
        ) as run:
            prompt = (
                "For each step below, return JSON "
                "[{step, automatable_0_to_1, rationale}]. "
                f"SOP: {sop['name']}\nSteps:\n"
                + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sop["steps"]))
            )
            raw = _run_engine(run, prompt, loops=loops)
            run.set_output(raw)
        run_ids.append(run.run_id)
    return run_ids
