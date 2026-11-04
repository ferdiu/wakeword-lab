"""Generate reproducible low-level synthetic backgrounds for wake-word negatives."""
from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
from scipy.io import wavfile


SAMPLE_RATE = 16_000


def _colored_noise(rng: np.random.Generator, length: int, profile: str) -> np.ndarray:
    white = rng.normal(0.0, 1.0, length)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(length, 1.0 / SAMPLE_RATE)
    floor = SAMPLE_RATE / length
    if profile == "white":
        shape = np.ones_like(frequencies)
    elif profile == "pink":
        shape = 1.0 / np.sqrt(np.maximum(frequencies, floor))
    elif profile == "brown":
        shape = 1.0 / np.maximum(frequencies, floor)
    elif profile == "blue":
        shape = np.sqrt(np.maximum(frequencies, floor))
    elif profile == "lowpass":
        shape = 1.0 / (1.0 + (frequencies / rng.uniform(250.0, 900.0)) ** 4)
    else:
        raise ValueError(f"Unknown synthetic noise profile: {profile}")
    colored = np.fft.irfft(spectrum * shape, n=length).astype(np.float32)
    colored -= float(colored.mean())
    rms = float(np.sqrt(np.mean(colored * colored) + 1e-12))
    if rms <= 1e-8:
        raise RuntimeError("Noise generator produced a near-silent clip")
    return colored / rms


def _make_clip(rng: np.random.Generator, profile: str) -> np.ndarray:
    length = SAMPLE_RATE
    noise = _colored_noise(rng, length, profile)
    t = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    if profile in ("brown", "lowpass"):
        # Add a quiet randomized hum/fan component to broaden low-frequency backgrounds.
        frequency = float(rng.choice([50.0, 60.0, 90.0, 120.0, 180.0, 240.0]))
        noise += float(rng.uniform(0.03, 0.18)) * np.sin(2.0 * np.pi * frequency * t + rng.uniform(0, 2*np.pi))
    if profile == "pink":
        # Slow modulation approximates a changing room/air-noise floor.
        mod_frequency = float(rng.uniform(0.2, 2.0))
        modulation = 0.75 + 0.25 * np.sin(2.0 * np.pi * mod_frequency * t + rng.uniform(0, 2*np.pi))
        noise *= modulation
    noise_rms = float(rng.uniform(0.0015, 0.035))
    noise *= noise_rms / (np.sqrt(np.mean(noise * noise)) + 1e-12)
    return np.clip(noise, -0.25, 0.25).astype(np.float32)


def generate_noise(project: str | Path, count: int = 400, seed: int = 42) -> list[Path]:
    """Write one-second synthetic noise-only WAVs under data/raw/noise/.

    Generated backgrounds are low-level colored noise and tonal room/fan/hum profiles.
    They are automatically assigned to the training partition by dataset preparation.
    Use genuine held-out room recordings in data/raw/negative for operational evaluation.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    project = Path(project)
    output = project / "data/raw/noise"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    profiles = ("white", "pink", "brown", "blue", "lowpass")
    run_id = uuid.uuid4().hex[:8]
    paths = []
    for index in range(count):
        profile = profiles[index % len(profiles)]
        clip = _make_clip(rng, profile)
        path = output / f"synthetic_{run_id}_{index:05d}_{profile}.wav"
        pcm = np.round(np.clip(clip, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(path, SAMPLE_RATE, pcm)
        paths.append(path)
    return paths
