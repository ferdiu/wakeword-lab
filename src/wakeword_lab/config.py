from __future__ import annotations

from pathlib import Path
import math
from typing import Any
import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "sample_rate": 16000,
    "clip_seconds": 1.0,
    "frame_length_ms": 25,
    "frame_step_ms": 10,
    "fft_length": 512,
    "mel_bins": 40,
    "lower_hz": 60.0,
    "upper_hz": 7600.0,
    "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "augmentation": {
        "enabled": True,
        "copies_per_positive": 10,
        "speed_min": 0.80,
        "speed_max": 1.20,
        "pitch_semitones": 4.0,
        "max_shift_ms": 200,
        "reverb_probability": 0.35,
        "noise_probability": 0.65,
        "snr_min_db": 5.0,
        "snr_max_db": 25.0,
    },
    "dataset": {"negative_hop_ms": 500, "max_windows_per_recording": 80, "noise_as_negative_train": True},
    "model": {
        "width": 24,
        "dropout": 0.15,
        "epochs": 40,
        "batch_size": 64,
        "learning_rate": 0.001,
        "label_smoothing": 0.05,
        "early_stopping_patience": 7,
    },
    "quantization": {"calibration_samples": 200, "target_validation_fpr": 0.001},
}


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if path is not None:
        with Path(path).open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a YAML mapping")
        config = _merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(c: dict) -> None:
    if c.get("sample_rate") != 16000 or c.get("clip_seconds") != 1.0:
        raise ValueError("The Android frontend contract requires 16 kHz, one-second clips")
    if c.get("frame_length_ms") != 25 or c.get("frame_step_ms") != 10:
        raise ValueError("The supported frontend requires 25 ms frames and 10 ms hop")
    if c.get("fft_length") != 512 or c.get("mel_bins") != 40:
        raise ValueError("The supported frontend requires a 512 FFT and 40 mel bins")
    if c.get("lower_hz") != 60.0 or c.get("upper_hz") != 7600.0:
        raise ValueError("The supported frontend requires mel bounds of 60-7600 Hz")
    split = c["split"]
    fractions = [float(split[k]) for k in ("train", "validation", "test")]
    if any(not 0 < value < 1 for value in fractions) or abs(sum(fractions) - 1) > 1e-7:
        raise ValueError("Split fractions must be in (0,1) and sum to 1")
    aug = c["augmentation"]
    if not isinstance(aug["enabled"], bool):
        raise ValueError("augmentation.enabled must be a YAML boolean")
    if int(aug["copies_per_positive"]) < 0:
        raise ValueError("copies_per_positive cannot be negative")
    speed_min, speed_max = float(aug["speed_min"]), float(aug["speed_max"])
    if not (0 < speed_min <= speed_max) or not (math.isfinite(speed_min) and math.isfinite(speed_max)):
        raise ValueError("Invalid speed range")
    probabilities = (float(aug["reverb_probability"]), float(aug["noise_probability"]))
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Augmentation probabilities must be finite and in [0,1]")
    pitch_range = float(aug["pitch_semitones"])
    shift_ms = int(aug["max_shift_ms"])
    snr_min, snr_max = float(aug["snr_min_db"]), float(aug["snr_max_db"])
    if not math.isfinite(pitch_range) or pitch_range < 0 or shift_ms < 0:
        raise ValueError("Pitch and temporal shift ranges must be finite and non-negative")
    if not (math.isfinite(snr_min) and math.isfinite(snr_max)) or snr_min > snr_max:
        raise ValueError("SNR bounds must be finite and snr_min_db cannot exceed snr_max_db")
    dataset = c["dataset"]
    if not isinstance(dataset.get("noise_as_negative_train", True), bool):
        raise ValueError("dataset.noise_as_negative_train must be a YAML boolean")
    if int(dataset["negative_hop_ms"]) < 1 or int(dataset["max_windows_per_recording"]) < 1:
        raise ValueError("Dataset hop and max_windows_per_recording must be positive")
    model = c["model"]
    if int(model["width"]) < 4 or int(model["epochs"]) < 1 or int(model["batch_size"]) < 1:
        raise ValueError("Invalid model training configuration")
    learning_rate = float(model["learning_rate"])
    dropout = float(model["dropout"])
    smoothing = float(model["label_smoothing"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(dropout) or not math.isfinite(smoothing) or not 0 <= dropout < 1 or not 0 <= smoothing < 0.5:
        raise ValueError("Invalid dropout or label smoothing")
    if int(c["quantization"]["calibration_samples"]) < 2:
        raise ValueError("calibration_samples must be at least two (one per class)")
    target_fpr = float(c["quantization"]["target_validation_fpr"])
    if not math.isfinite(target_fpr) or not 0 < target_fpr < 1:
        raise ValueError("target_validation_fpr must be finite and between 0 and 1")
