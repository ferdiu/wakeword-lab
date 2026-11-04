import numpy as np
from wakeword_lab.augment import augment_audio

def test_augmentation_is_deterministic_and_bounded():
    x = np.zeros(16000, np.float32); x[1000:7000] = .5
    a = augment_audio(x, np.random.default_rng(123), [], reverb_probability=1, noise_probability=0)
    b = augment_audio(x, np.random.default_rng(123), [], reverb_probability=1, noise_probability=0)
    assert a.shape == (16000,)
    assert np.isfinite(a).all() and np.max(np.abs(a)) <= 1
    assert np.array_equal(a, b)

def test_noise_augmentation_changes_waveform():
    x = np.ones(16000, np.float32) * .1
    noise = np.random.default_rng(4).normal(0, .1, 20000).astype(np.float32)
    y = augment_audio(x, np.random.default_rng(2), [noise], speed_min=1, speed_max=1,
                      pitch_semitones=0, max_shift_ms=0, reverb_probability=0,
                      noise_probability=1)
    assert not np.allclose(x, y)

def test_temporal_shift_is_bounded_to_window_even_for_large_config():
    x = np.zeros(16000, np.float32)
    x[100:200] = 0.5
    y = augment_audio(
        x, np.random.default_rng(9), [], speed_min=1, speed_max=1,
        pitch_semitones=0, max_shift_ms=5000, reverb_probability=0,
        noise_probability=0,
    )
    assert y.shape == (16000,)
    assert np.isfinite(y).all()
