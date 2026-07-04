"""Adapter for a real Hermes Agent (Nous Research) session.

Honesty note: this has not been run against a live Hermes gateway -- Nous's
CLI/gateway wire format is not something this harness can verify without
credentials. What's implemented is the *contract* the rest of the harness
depends on (drives a fresh subagent per session, exposes a debrief hook, and
exposes a skill directory for loop/skills.py to snapshot). Wire it to the
real gateway by adjusting `_call_gateway` below; the session orchestration,
debrief protocol, leakage lint, and skill diffing in orchestrator/ and loop/
are exercised by their own unit tests and don't depend on this adapter
existing to be correct -- only an actual Hermes-loop *demonstration* (a real
score-vs-session learning curve) needs this wired to a live gateway.

Injection boundary (honesty rule, see project brief Sec 5.8): this adapter
must never write, edit, or seed Hermes's skill files. It only reads the
skill directory for snapshotting and sends the debrief message.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import httpx

from agents.base import Reply

DEFAULT_GATEWAY_URL = "http://localhost:8008"


class HermesPilot:
    def __init__(
        self,
        gateway_url: Optional[str] = None,
        skill_directory: Optional[str] = None,
        timeout_s: float = 30.0,
    ):
        self.gateway_url = (gateway_url or os.environ.get("HERMES_GATEWAY_URL") or DEFAULT_GATEWAY_URL).rstrip("/")
        self._skill_directory = skill_directory or os.environ.get("HERMES_SKILL_DIR")
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)
        self._session_id: Optional[str] = None

    def start_session(self, role_prompt: str) -> None:
        resp = self._client.post(f"{self.gateway_url}/sessions", json={"system_prompt": role_prompt})
        resp.raise_for_status()
        self._session_id = resp.json()["session_id"]

    def on_transmission(self, transmission: str) -> Reply:
        if not self._session_id:
            raise RuntimeError("start_session() must be called before on_transmission()")
        t0 = time.monotonic()
        resp = self._client.post(
            f"{self.gateway_url}/sessions/{self._session_id}/messages",
            json={"role": "user", "content": transmission},
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = resp.json()["content"].strip()
        return Reply(text=text, latency_ms=latency_ms)

    def end_session(self) -> None:
        if self._session_id:
            self._client.post(f"{self.gateway_url}/sessions/{self._session_id}/end")
        self._session_id = None

    def debrief(self, scorecard: dict) -> None:
        """Observational-only scorecard message. loop/debrief.py builds the
        actual text and runs the leakage lint before this is ever called."""
        if not self._session_id:
            raise RuntimeError("no active session to debrief")
        resp = self._client.post(
            f"{self.gateway_url}/sessions/{self._session_id}/messages",
            json={"role": "user", "content": scorecard.get("debrief_text", "")},
        )
        resp.raise_for_status()

    def skill_dir(self) -> Optional[str]:
        return self._skill_directory
