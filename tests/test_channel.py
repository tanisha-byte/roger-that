import numpy as np

from channel.dsp import degrade, measure_snr_db, SNR_TIERS
from channel.tts import SyntheticTTS


def test_synthetic_tts_produces_audio():
    tts = SyntheticTTS()
    audio, sr = tts.synthesize("descend flight level eight zero")
    assert sr > 0
    assert len(audio) > 0
    assert audio.dtype == np.float32


def test_degrade_all_tiers_run_and_lengthen_for_squelch_tail():
    tts = SyntheticTTS()
    audio, sr = tts.synthesize("victor tango alpha bravo charlie")
    for tier in SNR_TIERS:
        out = degrade(audio, sr, tier, seed=1)
        assert len(out) > len(audio)
        assert np.isfinite(out).all()


def test_lower_snr_tier_is_measurably_noisier():
    tts = SyntheticTTS()
    audio, sr = tts.synthesize("qnh one zero one three")
    snr_20 = measure_snr_db(audio, degrade(audio, sr, "20db", seed=2))
    snr_5 = measure_snr_db(audio, degrade(audio, sr, "5db", seed=2))
    assert snr_20 > snr_5


def test_unknown_tier_rejected():
    tts = SyntheticTTS()
    audio, sr = tts.synthesize("hello")
    try:
        degrade(audio, sr, "not-a-tier")
        assert False, "expected ValueError"
    except ValueError:
        pass
