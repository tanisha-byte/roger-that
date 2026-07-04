"""Radio channel degradation DSP chain.

300-3400 Hz bandpass -> soft-clip/compand -> additive noise at a target SNR
-> squelch tail on transmission end. This operates on any float32 PCM
buffer regardless of what produced it, so it's testable with a synthetic
tone and doesn't require a live TTS provider (see channel/tts.py).

SNR tiers per the project brief: clean / 20 dB / 15 dB / 10 dB / 8 dB / 5 dB.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter

SNR_TIERS = {"clean": None, "20db": 20.0, "15db": 15.0, "10db": 10.0, "8db": 8.0, "5db": 5.0}

BANDPASS_LOW_HZ = 300.0
BANDPASS_HIGH_HZ = 3400.0
SQUELCH_TAIL_S = 0.12


def bandpass(audio: np.ndarray, sr: int, low: float = BANDPASS_LOW_HZ, high: float = BANDPASS_HIGH_HZ, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, audio).astype(np.float32)


def soft_clip(audio: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """Companding/clipping stand-in: a tanh soft-clipper, which is what
    cheap radio transmitters effectively do to a hot signal."""
    return np.tanh(audio * drive).astype(np.float32) / np.tanh(drive)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def add_noise_at_snr(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_rms = _rms(audio)
    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_rms = _rms(noise)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise *= (target_noise_rms / (noise_rms + 1e-12))
    return (audio + noise).astype(np.float32)


def squelch_tail(sr: int, duration_s: float = SQUELCH_TAIL_S, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = int(sr * duration_s)
    burst = rng.standard_normal(n).astype(np.float32) * 0.15
    envelope = np.linspace(1.0, 0.0, n, dtype=np.float32) ** 2
    return burst * envelope


def degrade(audio: np.ndarray, sr: int, snr_tier: str, seed: int = 0) -> np.ndarray:
    if snr_tier not in SNR_TIERS:
        raise ValueError(f"unknown SNR tier: {snr_tier}. Choose from {list(SNR_TIERS)}")
    rng = np.random.default_rng(seed)

    out = bandpass(audio.astype(np.float32), sr)
    snr_db = SNR_TIERS[snr_tier]
    if snr_db is not None:
        out = soft_clip(out)
        out = add_noise_at_snr(out, snr_db, rng)

    tail = squelch_tail(sr, rng=rng)
    return np.concatenate([out, tail]).astype(np.float32)


def measure_snr_db(clean: np.ndarray, degraded: np.ndarray) -> float:
    """For golden-testing the chain itself: recovers an approximate
    achieved SNR by comparing signal RMS to the residual after subtracting
    the (length-aligned) clean signal."""
    n = min(len(clean), len(degraded))
    residual = degraded[:n] - clean[:n]
    signal_rms = _rms(clean[:n])
    residual_rms = _rms(residual)
    if residual_rms == 0:
        return float("inf")
    return 20 * np.log10(signal_rms / residual_rms)
