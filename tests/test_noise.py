import numpy as np
import pytest
from scipy.io import wavfile
from wakeword_lab.noise import generate_noise


def test_generate_noise_produces_bounded_non_silent_16khz_wavs(tmp_path):
    paths = generate_noise(tmp_path, count=7, seed=123)
    assert len(paths) == 7
    assert len(list((tmp_path / "data/raw/noise").glob("*.wav"))) == 7
    for path in paths:
        rate, samples = wavfile.read(path)
        assert rate == 16000 and samples.shape == (16000,)
        assert samples.dtype == np.int16
        assert 0.0008 < float(np.sqrt(np.mean((samples.astype(np.float32) / 32768) ** 2))) < 0.04


def test_generate_noise_requires_positive_count(tmp_path):
    with pytest.raises(ValueError):
        generate_noise(tmp_path, count=0)
