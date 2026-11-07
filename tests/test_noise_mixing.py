import csv

import numpy as np
import pytest
from scipy.io import wavfile

from wakeword_lab import dataset
from wakeword_lab.augment import mix_with_noise
from wakeword_lab.dataset import prepare

SPLITS = ("train", "validation", "test")


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


def test_mix_with_noise_hits_target_snr():
    rng = np.random.default_rng(3)
    window = (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    noise = rng.normal(0.0, 0.02, 16000).astype(np.float32)
    snr_db = 12.5
    mixed = mix_with_noise(window, noise, snr_db)
    achieved = 20.0 * np.log10(_rms(window) / _rms(mixed - window))
    assert abs(achieved - snr_db) < 1e-2
    assert mixed.dtype == np.float32 and len(mixed) == 16000


def test_mix_with_noise_pads_short_noise_to_window_length():
    window = np.full(16000, 0.1, np.float32)
    noise = np.full(8000, 0.05, np.float32)
    mixed = mix_with_noise(window, noise, 6.0)
    assert len(mixed) == 16000
    achieved = 20.0 * np.log10(_rms(window) / _rms(mixed - window))
    assert abs(achieved - 6.0) < 1e-2


def test_mix_with_noise_guards_zero_rms_window():
    silence = np.zeros(16000, np.float32)
    noise = np.full(16000, 0.05, np.float32)
    mixed = mix_with_noise(silence, noise, 10.0)
    assert not np.isnan(mixed).any()
    assert np.array_equal(mixed, silence)


def test_mix_with_noise_guards_zero_rms_noise():
    window = np.full(16000, 0.1, np.float32)
    mixed = mix_with_noise(window, np.zeros(16000, np.float32), 10.0)
    assert not np.isnan(mixed).any()
    assert np.array_equal(mixed, window)


def _wav_constant(path, value, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.full(int(16000 * seconds), value, np.float32)
    wavfile.write(path, 16000, (x * 32767).astype(np.int16))


# Distinct constant amplitude per noise group: the mean of a mixed snippet
# identifies which partition's noise group it came from.
NOISE_SIGNATURE = {"train": 0.05, "validation": 0.06, "test": 0.07}
COPIES = 2


def _project(tmp_path):
    raw = tmp_path / "data/raw"
    for index in range(12):
        _wav_constant(raw / "positive" / f"p{index}__speaker{index}.wav", 0.2)
        _wav_constant(raw / "negative" / f"n{index}__room{index}.wav", 0.01)
    for split, signature in NOISE_SIGNATURE.items():
        _wav_constant(raw / "noise" / f"demand_{split}__ambient.wav", signature)
    with (raw / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "group_id", "split"])
        for split in SPLITS:
            writer.writerow([f"demand_{split}__ambient.wav", f"demand_{split}", split])
    (tmp_path / "wakeword.yaml").write_text(
        "dataset:\n  noise_as_negative_train: false\n"
        "noise_mixing:\n  enabled: true\n  copies_per_clip: 2\n"
        "  snr_min_db: 5.0\n  snr_max_db: 15.0\n",
        encoding="utf-8")
    return tmp_path


def test_prepare_mixes_noise_into_every_split_without_leakage(tmp_path, monkeypatch):
    project = _project(tmp_path)
    used_noise_means = []
    real_mix = dataset.mix_with_noise

    def spy(window, noise, snr_db):
        used_noise_means.append(float(np.mean(noise)))
        return real_mix(window, noise, snr_db)

    monkeypatch.setattr(dataset, "mix_with_noise", spy)
    manifest = prepare(project, seed=1, augment=False)

    # Originals are preserved: every windowed clip yields copies_per_clip extra
    # mixed copies carrying the same label, in every partition.
    for split in SPLITS:
        summary = manifest["summary"][split]
        windows = summary["examples"] - summary["noise_mixed"]
        assert windows > 0
        assert summary["noise_mixed"] == COPIES * windows
        assert manifest["noise_mixing"]["enabled"] is True
    # Window count equals the positive+negative rows the split received.
    for split in SPLITS:
        windows = sum(1 for row in manifest["splits"][split] if row["kind"] != "noise")
        assert manifest["summary"][split]["examples"] - manifest["summary"][split]["noise_mixed"] == windows

    # Splits are processed in order, so the recorded snippet means slice by
    # per-split mixed counts; each snippet must come from that same partition's
    # own noise group (no cross-split leakage).
    offset = 0
    for split in SPLITS:
        count = manifest["summary"][split]["noise_mixed"]
        used = used_noise_means[offset:offset + count]
        offset += count
        assert len(used) == count
        for mean in used:
            assert abs(mean - NOISE_SIGNATURE[split]) < 0.01
    assert offset == len(used_noise_means)


def test_prepare_noise_mixing_is_seed_deterministic(tmp_path):
    manifests = []
    arrays = []
    for run in ("a", "b"):
        project = _project(tmp_path / run)
        manifests.append(prepare(project, seed=7, augment=False))
        arrays.append({split: np.load(project / f"data/processed/{split}.npz")["x"]
                       for split in SPLITS})
    assert manifests[0]["summary"] == manifests[1]["summary"]
    for split in SPLITS:
        assert np.array_equal(arrays[0][split], arrays[1][split])


def test_prepare_warns_when_partition_has_no_noise_groups(tmp_path):
    raw = tmp_path / "data/raw"
    for index in range(12):
        _wav_constant(raw / "positive" / f"p{index}__speaker{index}.wav", 0.2)
        _wav_constant(raw / "negative" / f"n{index}__room{index}.wav", 0.01)
    (tmp_path / "wakeword.yaml").write_text("noise_mixing:\n  enabled: true\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="has no noise groups"):
        prepare(tmp_path, seed=1, augment=False)
    # Mixing is skipped per partition, so no mixed copies exist anywhere.
    manifest = prepare(tmp_path, seed=1, augment=False)
    for split in SPLITS:
        assert manifest["summary"][split]["noise_mixed"] == 0
