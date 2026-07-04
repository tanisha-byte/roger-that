"""Pluggable reference ASR for the agent's replies in voice mode. Fixed,
documented model per campaign, so agent-side scoring is never confounded by
the harness's own transcription (per the project brief's risk mitigation
for ASR confusion).

No reference ASR ships configured in this workspace -- there's no audio
model to bundle. `NullASR` fails loudly rather than silently returning
empty text, which would otherwise look identical to a real "pilot said
nothing" case and corrupt scoring.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class ASRProvider(Protocol):
    def transcribe(self, audio: np.ndarray, sr: int) -> str: ...


class NullASR:
    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        raise NotImplementedError(
            "no reference ASR is configured -- voice-mode scoring requires a fixed, documented ASR model "
            "(see project brief Sec 5.3). Wire one up here (e.g. Whisper) before running voice-mode campaigns."
        )


class WhisperAPIASR:
    """Structural only -- not exercised in this workspace (no credentials).
    Wire up to `POST /v1/audio/transcriptions`."""

    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        raise NotImplementedError("wire this up to your ASR provider's API; see NullASR for the interface shape")
