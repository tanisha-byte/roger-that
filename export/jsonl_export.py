"""JSONL trajectory exporter: (role_prompt, transmissions[], replies[],
per_turn_scores[], session_reward) per session, read straight out of the
sessions/turns tables so it's guaranteed to match what the dashboard shows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agents.base import ROLE_PROMPT_TEMPLATE
from db.store import Store


def export_jsonl(store: Store, out_path: str, campaign_id: Optional[str] = None, limit: int = 10_000) -> int:
    sessions = store.list_sessions(campaign_id=campaign_id, limit=limit)
    count = 0
    with open(out_path, "w") as f:
        for s in sessions:
            full = store.get_session(s["id"])
            transmissions = [t["atc_text"] for t in full["turns"]]
            replies = [t["reply_text"] for t in full["turns"]]
            per_turn_scores = [t["record"] for t in full["turns"]]
            record = {
                "session_id": full["id"],
                "campaign_id": full["campaign_id"],
                "agent": full["agent"],
                "scenario_id": full["scenario_id"],
                "mode": full["mode"],
                "snr": full["snr"],
                "pool": full["pool"],
                "role_prompt": ROLE_PROMPT_TEMPLATE.format(callsign="<redacted-per-session>"),
                "transmissions": transmissions,
                "replies": replies,
                "per_turn_scores": per_turn_scores,
                "session_reward": full["score"],
                "scorecard": full["scorecard"],
            }
            f.write(json.dumps(record) + "\n")
            count += 1
    return count
