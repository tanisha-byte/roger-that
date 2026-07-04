"""FastAPI + dashboard, one port.

Endpoints match the project brief Sec 5.10:
  GET  /api/leaderboard?metric=score&snr=15
  GET  /api/agents/{id}/curve
  GET  /api/sessions/{id}
  GET  /api/sessions/{id}/audio/{turn}
  GET  /api/skills/{agent}/timeline
  POST /api/campaigns
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.openai_compat import OpenAICompatPilot
from db.store import DEFAULT_DB_PATH, Store
from orchestrator import list_scenarios, run_campaign

app = FastAPI(title="Roger That")

DASHBOARD_DIR = Path(__file__).parent / "dashboard"


def _has_llm_key() -> bool:
    return bool(os.environ.get("ROGER_THAT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _has_bolna_key() -> bool:
    if os.environ.get("BOLNA_API_KEY"):
        return True
    env_file = Path(__file__).parent / ".env"
    return env_file.exists() and "BOLNA_API_KEY=" in env_file.read_text()


@app.get("/api/config")
def config():
    """Reports which real capabilities are actually available, so the
    dashboard can be honest about what a button will do instead of a mock
    silently standing in. No agent runs against fabricated behavior here --
    if a key isn't configured, the corresponding feature says so."""
    return {
        "llm_campaigns_available": _has_llm_key(),
        "bolna_live_calls_available": _has_bolna_key(),
    }


def get_store() -> Store:
    return Store(db_path=DEFAULT_DB_PATH)


@app.get("/api/leaderboard")
def leaderboard(metric: str = "score", snr: Optional[str] = None):
    store = get_store()
    try:
        return store.leaderboard(metric=metric, snr=snr)
    finally:
        store.close()


@app.get("/api/agents/{agent_id}/curve")
def agent_curve(agent_id: str):
    store = get_store()
    try:
        return store.agent_curve(agent_id)
    finally:
        store.close()


@app.get("/api/sessions")
def sessions(campaign_id: Optional[str] = None, limit: int = 100):
    store = get_store()
    try:
        return store.list_sessions(campaign_id=campaign_id, limit=limit)
    finally:
        store.close()


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: int):
    store = get_store()
    try:
        session = store.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session
    finally:
        store.close()


@app.get("/api/sessions/{session_id}/audio/{turn}")
def session_audio(session_id: int, turn: int):
    # Voice mode is not wired to a live TTS/ASR provider in this workspace
    # (see channel/README notes) -- text-mode sessions have no audio to
    # serve. Kept as a real 501 rather than a fake 200 with empty bytes.
    raise HTTPException(status_code=501, detail="voice-mode audio archival requires a configured TTS provider; see channel/tts.py")


@app.get("/api/skills/{agent_id}/timeline")
def skill_timeline(agent_id: str):
    store = get_store()
    try:
        return store.skill_timeline(agent_id)
    finally:
        store.close()


@app.get("/api/scenarios")
def scenarios():
    return [{"id": s["id"], "title": s.get("title", s["id"]), "difficulty": s.get("difficulty", 1)} for s in list_scenarios()]


class CampaignRequest(BaseModel):
    campaign_id: str
    agent_name: str
    model: str  # any OpenAI-compatible chat-completions model id
    n_sessions: int = 20
    base_seed: int = 1
    held_out_scenario_ids: list[str] = []


def _run_campaign_job(req: CampaignRequest):
    store = get_store()
    try:
        all_scenarios = list_scenarios()
        held_out = [s for s in all_scenarios if s["id"] in req.held_out_scenario_ids] or all_scenarios[-1:]
        train = [s for s in all_scenarios if s not in held_out] or all_scenarios
        run_campaign(
            store=store,
            campaign_id=req.campaign_id,
            agent_name=req.agent_name,
            agent_factory=lambda i: OpenAICompatPilot(model=req.model),
            train_scenarios=train,
            held_out_scenarios=held_out,
            n_sessions=req.n_sessions,
            base_seed=req.base_seed,
        )
    finally:
        store.close()


@app.post("/api/campaigns")
def launch_campaign(req: CampaignRequest, background_tasks: BackgroundTasks):
    """Runs a real campaign against a real OpenAI-compatible model -- there
    is no mock fallback. Refuses to start (rather than fail silently deep in
    a background task) if no key is configured."""
    if not _has_llm_key():
        raise HTTPException(
            status_code=400,
            detail="no LLM key configured -- set ROGER_THAT_LLM_API_KEY or OPENAI_API_KEY before launching a campaign",
        )
    background_tasks.add_task(_run_campaign_job, req)
    return JSONResponse({"status": "started", "campaign_id": req.campaign_id, "agent_name": req.agent_name})


app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
