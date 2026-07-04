"""Session orchestrator: runs campaigns of (agent, scenario_pool, n_sessions,
seed), enforcing the train/held-out split -- debriefs only go out for
training-pool sessions; held-out sessions are scored silently. This is the
backstop against an agent "memorizing despite randomization" (see project
brief Sec 12): even if scenario randomization somehow leaked, held-out
sessions never receive a debrief, so a real skill/procedure gap would still
show up on the held-out curve.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from agents.base import PilotAgent
from db.store import Store
from loop.debrief import DebriefLeakageError, send_debrief
from loop.skills import read_skill_dir
from orchestrator.session import run_session

HELD_OUT_EVERY = 4  # ~1 held-out session per 3 training sessions, per brief's 30/10 ratio


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_campaign(
    store: Store,
    campaign_id: str,
    agent_name: str,
    agent_factory: Callable[[int], PilotAgent],
    train_scenarios: List[dict],
    held_out_scenarios: List[dict],
    n_sessions: int,
    base_seed: int,
    mode: str = "text",
    snr: Optional[str] = None,
    on_session_complete: Optional[Callable[[int, dict], None]] = None,
) -> List[Dict[str, Any]]:
    if not train_scenarios:
        raise ValueError("train_scenarios must be non-empty")
    if not held_out_scenarios:
        held_out_scenarios = train_scenarios

    scorecards: List[Dict[str, Any]] = []

    for session_index in range(n_sessions):
        is_held_out = (session_index + 1) % HELD_OUT_EVERY == 0
        pool = "held_out" if is_held_out else "train"
        pool_list = held_out_scenarios if is_held_out else train_scenarios
        scenario = pool_list[session_index % len(pool_list)]
        seed = base_seed + session_index

        agent = agent_factory(session_index)
        result = run_session(agent, scenario, seed)
        scorecard = result.scorecard

        session_id = store.insert_session({
            "campaign_id": campaign_id,
            "agent": agent_name,
            "scenario_id": scenario["id"],
            "seed": seed,
            "mode": mode,
            "snr": snr,
            "pool": pool,
            "session_index": session_index,
            "score": scorecard["score"],
            "readback_completeness": scorecard["readback_completeness"],
            "callsign_discipline": scorecard["callsign_discipline"],
            "safety_score": scorecard["safety_score"],
            "safety_violations": scorecard["safety_violations"],
            "hearback_catch_rate": scorecard["hearback_catch_rate"],
            "state_consistency": scorecard["state_consistency"],
            "emergency_completeness": scorecard["emergency_completeness"],
            "weights_version": scorecard["weights_version"],
            "scorecard_json": _to_json_safe(scorecard),
            "created_at": _now(),
        })
        store.insert_turns(session_id, result.turn_rows)

        if pool == "train" and hasattr(agent, "debrief"):
            try:
                send_debrief(agent, scorecard)
            except DebriefLeakageError as e:
                print(f"[campaign] debrief leakage blocked for session {session_index}: {e}")

        if hasattr(agent, "skill_dir"):
            skill_dir = agent.skill_dir()
            if skill_dir:
                for filename, content in read_skill_dir(skill_dir).items():
                    store.insert_skill_snapshot(agent_name, session_index, filename, content, _now())

        scorecard_with_meta = {**scorecard, "session_id": session_id, "session_index": session_index, "pool": pool, "scenario_id": scenario["id"]}
        scorecards.append(scorecard_with_meta)
        if on_session_complete:
            on_session_complete(session_index, scorecard_with_meta)

    return scorecards


def _to_json_safe(scorecard: dict) -> str:
    import json

    return json.dumps(scorecard)
