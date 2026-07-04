"""Pluggable TTS for the controller voice. Any provider works as long as it
returns (float32 PCM, sample_rate); the controller voice is fixed per
campaign for comparability, per the project brief.

`SyntheticTTS` needs no credentials and no network access -- it renders
each word as a short tone burst (pitch varies with a hash of the word so
different words are distinguishable in a waveform, though obviously not
intelligible speech). It exists purely so channel/dsp.py and the
voice-mode plumbing can be exercised end-to-end in this workspace, which
has no TTS credentials configured. Swap in a real provider for anything
resembling an actual voice-mode benchmark.
"""
from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np


class TTSProvider(Protocol):
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]: ...


class SyntheticTTS:
    def __init__(self, sr: int = 16000, wpm: float = 160.0):
        self.sr = sr
        self.word_duration_s = 60.0 / wpm

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        words = text.split() or [""]
        chunks = [self._tone_for_word(w) for w in words]
        gap = np.zeros(int(self.sr * 0.03), dtype=np.float32)
        audio = np.concatenate([c for pair in zip(chunks, [gap] * len(chunks)) for c in pair])
        return audio.astype(np.float32), self.sr

    def _tone_for_word(self, word: str) -> np.ndarray:
        freq = 150 + (hash(word) % 250)  # deterministic per-word pitch, human-voice-ish range
        n = int(self.sr * self.word_duration_s)
        t = np.linspace(0, self.word_duration_s, n, endpoint=False)
        envelope = np.hanning(n).astype(np.float32)
        return (0.6 * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)


class OpenAITTS:
    """Structural only -- not exercised in this workspace (no credentials).
    Wire up to `POST /v1/audio/speech` and return the decoded PCM."""

    def __init__(self, api_key: str, voice: str = "onyx", model: str = "tts-1"):
        self.api_key = api_key
        self.voice = voice
        self.model = model

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        raise NotImplementedError("wire this up to your TTS provider's API; see SyntheticTTS for the interface shape")
