"""Generic audio-in/audio-out websocket adapter, for Bolna-style voice
pipelines and any other websocket voice agent.

Honesty note: like agents/hermes.py, this is a contract-level implementation
-- it speaks a simple documented JSON-over-websocket protocol (base64 audio
frames in, base64 audio frames + a final transcript out) rather than any one
vendor's actual wire format, since no live voice endpoint was available to
test against. Point `ws_url` at a real endpoint that speaks this protocol,
or adapt `_send_audio`/`_recv_reply` to match the target vendor.

Voice-mode session flow (see channel/ for the DSP chain this plugs into):
  1. orchestrator TTS-synthesizes the controller line and degrades it
     (channel/tts.py + channel/dsp.py)
  2. this adapter sends the degraded audio over the websocket
  3. the agent-under-test's own pipeline (its ASR/LLM/TTS) produces audio
  4. this adapter receives that audio and hands it to the harness's
     reference ASR (channel/asr.py) for scoring -- so agent-side scoring is
     never confounded by the harness's own transcription of what the
     *controller* said, only by the one fixed reference model used
     uniformly across all agents.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Optional

from agents.base import Reply

try:
    from websockets.sync.client import connect as ws_connect
except ImportError:  # pragma: no cover - optional dependency, voice mode only
    ws_connect = None


class WebsocketVoicePilot:
    def __init__(self, ws_url: str, reference_asr=None, timeout_s: float = 30.0):
        if ws_connect is None:
            raise RuntimeError(
                "the 'websockets' package is required for voice-mode agents; "
                "install it or use the text-mode openai_compat adapter instead"
            )
        self.ws_url = ws_url
        self.reference_asr = reference_asr
        self.timeout_s = timeout_s
        self._ws = None

    def start_session(self, role_prompt: str) -> None:
        self._ws = ws_connect(self.ws_url, open_timeout=self.timeout_s)
        self._ws.send(json.dumps({"type": "session_start", "role_prompt": role_prompt}))

    def on_transmission(self, transmission) -> Reply:
        """`transmission` is raw PCM/WAV bytes in voice mode (the degraded
        controller audio); text mode never routes through this adapter."""
        t0 = time.monotonic()
        payload = transmission if isinstance(transmission, (bytes, bytearray)) else str(transmission).encode()
        self._ws.send(json.dumps({"type": "audio", "data": base64.b64encode(payload).decode()}))
        raw = self._ws.recv(timeout=self.timeout_s)
        msg = json.loads(raw)
        latency_ms = int((time.monotonic() - t0) * 1000)
        audio_bytes = base64.b64decode(msg["data"]) if msg.get("data") else b""

        text = msg.get("transcript", "")
        if not text and self.reference_asr is not None and audio_bytes:
            text = self.reference_asr.transcribe(audio_bytes)

        return Reply(text=text, latency_ms=latency_ms, audio=audio_bytes)

    def end_session(self) -> None:
        if self._ws is not None:
            self._ws.send(json.dumps({"type": "session_end"}))
            self._ws.close()
            self._ws = None
