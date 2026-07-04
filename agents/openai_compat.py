"""Adapter for any OpenAI-compatible chat-completions endpoint (OpenAI,
OpenRouter, Azure OpenAI, local vLLM/Ollama servers with an openai-compat
route, etc). Covers most models under test via a single adapter.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional

import httpx

from agents.base import Reply

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatPilot:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        timeout_s: float = 20.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ROGER_THAT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("ROGER_THAT_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._messages: List[dict] = []
        self._client = httpx.Client(timeout=timeout_s)

    def start_session(self, role_prompt: str) -> None:
        self._messages = [{"role": "system", "content": role_prompt}]

    def on_transmission(self, transmission: str) -> Reply:
        self._messages.append({"role": "user", "content": transmission})
        t0 = time.monotonic()
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": self._messages, "temperature": self.temperature},
        )
        resp.raise_for_status()
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        self._messages.append({"role": "assistant", "content": text})
        return Reply(text=text, latency_ms=latency_ms)

    def end_session(self) -> None:
        self._client.close()
