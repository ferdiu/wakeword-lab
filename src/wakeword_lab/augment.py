from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly, fftconvolve
from math import gcd

from .audio import fit_window


def _resample_length(x: np.ndarray, out_len: int) -> np.ndarray:
    if len(x) == out_len:
        return x.astype(np.float32, copy=True)
    if not len(x):
        return np.zeros(out_len, np.float32)
    g = gcd(len(x), out_len)
    y = resample_poly(x, out_len // g, len(x) // g).astype(np.float32)
    return fit_window(y, out_len)


def augment_audio(
    audio: np.ndarray,
    rng: np.random.Generator,
    noises: list[np.ndarray] | None = None,
    *,
    sample_rate: int = 16000,
    speed_min: float = 0.8,
    speed_max: float = 1.2,
    pitch_semitones: float = 4.0,
    max_shift_ms: int = 200,
    reverb_probability: float = 0.35,
    noise_probability: float = 0.65,
    snr_min_db: float = 5.0,
    snr_max_db: float = 25.0,
) -> np.ndarray:
    """Apply a randomized speech-domain augmentation chain, returning one second of mono PCM.

    Pitch shifting uses librosa when installed (the optional training extra). Its fallback is a
    simple resampling pitch shift; that fallback is intentionally documented as approximate.
    """
    target_len = sample_rate
    x = fit_window(np.asarray(audio, np.float32), target_len)

    # Speed changes duration and pitch together, matching natural speaking-rate variation.
    speed = float(rng.uniform(speed_min, speed_max))
    if abs(speed - 1.0) > 1e-3:
        target = max(1, round(len(x) / speed))
        x = _resample_length(x, target)
        x = fit_window(x, target_len)

    # Pitch-shift approximately independently of duration where librosa is available.
    if pitch_semitones > 0:
        steps = float(rng.uniform(-pitch_semitones, pitch_semitones))
        if abs(steps) > 0.05:
            try:
                import librosa
                x = librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=steps).astype(np.float32)
            except ImportError:
                factor = 2.0 ** (steps / 12.0)
                shifted = _resample_length(x, max(1, round(len(x) / factor)))
                # Preserve the altered pitch; stretching back with another resample would
                # cancel the pitch change. Pad/crop to the model's fixed one-second window.
                x = fit_window(shifted, target_len)

    # Random temporal displacement; zero-fill rather than circularly wrapping speech.
    max_shift = min(round(sample_rate * max_shift_ms / 1000), target_len - 1)
    if max_shift:
        shift = int(rng.integers(-max_shift, max_shift + 1))
        shifted = np.zeros_like(x)
        if shift >= 0:
            shifted[shift:] = x[: len(x) - shift]
        else:
            shifted[:shift] = x[-shift:]
        x = shifted

    # A short exponentially decaying room impulse response.
    if rng.random() < reverb_probability:
        ir_len = int(rng.uniform(0.06, 0.22) * sample_rate)
        t = np.arange(ir_len, dtype=np.float32) / sample_rate
        decay = float(rng.uniform(8.0, 30.0))
        ir = np.exp(-decay * t).astype(np.float32)
        ir[0] = 1.0
        ir *= float(rng.uniform(0.08, 0.28))
        ir[0] = 1.0
        wet = fftconvolve(x, ir, mode="full")[:target_len].astype(np.float32)
        x = 0.78 * x + 0.22 * wet

    # Real ambient noise is sampled only from the training noise directory.
    if noises and rng.random() < noise_probability:
        noise = np.asarray(noises[int(rng.integers(0, len(noises)))], np.float32)
        if len(noise) >= target_len:
            start = int(rng.integers(0, len(noise) - target_len + 1))
            noise = noise[start : start + target_len]
        else:
            noise = np.resize(noise, target_len)
        signal_rms = float(np.sqrt(np.mean(x * x) + 1e-10))
        noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-10))
        if noise_rms > 1e-6 and signal_rms > 1e-6:
            snr = float(rng.uniform(snr_min_db, snr_max_db))
            noise *= signal_rms / (noise_rms * (10.0 ** (snr / 20.0)))
            x = x + noise

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def mix_with_noise(window: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix a one-second window with a noise snippet at a target SNR in dB.

    The noise is scaled so its RMS sits ``snr_db`` below the window RMS. Shorter
    noise snippets are zero-padded to the window length. Two degenerate cases are
    guarded instead of amplified: a zero-RMS window (silence) has no reference
    level and is returned unchanged, and a zero-RMS noise snippet contributes
    nothing. The mix is clipped to the PCM range like augment_audio output.
    """
    window = np.asarray(window, dtype=np.float32)
    noise = fit_window(np.asarray(noise, dtype=np.float32), len(window))
    signal_rms = float(np.sqrt(np.mean(window * window) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    if signal_rms <= 1e-6 or noise_rms <= 1e-6:
        return window.copy()
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    mixed = window + noise * (target_noise_rms / noise_rms)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)
