from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import warnings
import numpy as np
from scipy.io import wavfile

from .audio import mono_resample, fit_window
from .augment import augment_audio, mix_with_noise
from .config import load_config
from .frontend import log_mel
from .split import group_split


def read_wav(path: str | Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    x = np.asarray(data)
    if x.dtype.kind == "u":
        # PCM WAV with 8-bit samples is unsigned and centered at 128, unlike
        # 16/24/32-bit PCM. Convert it to signed float audio before downmixing.
        info = np.iinfo(x.dtype)
        midpoint = (info.max + 1) / 2.0
        x = (x.astype(np.float32) - midpoint) / midpoint
    elif x.dtype.kind == "i":
        info = np.iinfo(x.dtype)
        scale = float(max(abs(info.min), info.max))
        x = x.astype(np.float32) / scale
    else:
        x = x.astype(np.float32)
    return mono_resample(x, int(sr))


def _metadata(project: Path) -> dict[str, dict]:
    path = project / "data/raw/metadata.csv"
    if not path.exists():
        return {}
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = Path(row.get("path", "")).name
            if not name:
                continue
            result[name] = row
    return result


def scan(project: str | Path) -> list[dict]:
    """Discover WAVs. Metadata may include speaker_id, group_id, split and phrase times."""
    project = Path(project)
    metadata = _metadata(project)
    rows = []
    for folder, label in (("positive", 1), ("negative", 0), ("noise", None)):
        directory = project / "data/raw" / folder
        if not directory.exists():
            continue
        for file in sorted(directory.glob("*.wav")):
            meta = metadata.get(file.name, {})
            # group_id is an optional split unit (e.g. recording session). By default,
            # use speaker_id for strict speaker-independent splits; a session group_id
            # allows personalized same-speaker train/dev/test recordings without
            # splitting clips from one capture session across partitions.
            source_group = ((meta.get("group_id") or "").strip()
                            or (meta.get("speaker_id") or "").strip()
                            or file.stem.split("__")[0])
            split = (meta.get("split") or "").strip() or None
            if split not in (None, "train", "validation", "test"):
                raise ValueError(f"Invalid split {split!r} for {file.name}")
            phrase_start = (meta.get("phrase_start_s") or "").strip()
            phrase_end = (meta.get("phrase_end_s") or "").strip()
            if bool(phrase_start) != bool(phrase_end):
                raise ValueError(f"Both phrase_start_s and phrase_end_s are required for {file.name}")
            rows.append({
                "path": str(file), "label": label, "kind": folder,
                "speaker_id": (meta.get("speaker_id") or "").strip() or None,
                "source_group": source_group, "split": split,
                "locale": (meta.get("locale") or "").strip(),
                "phrase_start_s": float(phrase_start) if phrase_start else None,
                "phrase_end_s": float(phrase_end) if phrase_end else None,
            })
    return rows


def _positive_clips(
    audio: np.ndarray,
    rng: np.random.Generator,
    limit: int,
    phrase_start_s: float | None = None,
    phrase_end_s: float | None = None,
) -> list[np.ndarray]:
    """Create positive windows that actually contain the annotated wake phrase.

    Without phrase timestamps, a long recording is treated as one centered phrase capture,
    not dozens of arbitrary positive windows that could contain only unrelated partial speech.
    With timestamps, sample several legal one-second contexts that contain the entire phrase.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if (phrase_start_s is None) != (phrase_end_s is None):
        raise ValueError("Both phrase_start_s and phrase_end_s are required")

    # Validate annotation even for recordings <= one second. Otherwise malformed
    # timestamps would silently bypass validation via the short-clip fast path.
    start = end = None
    if phrase_start_s is not None:
        start_s, end_s = float(phrase_start_s), float(phrase_end_s)
        if not (math.isfinite(start_s) and math.isfinite(end_s)):
            raise ValueError("Wake phrase timestamps must be finite")
        start = round(start_s * 16000)
        end = round(end_s * 16000)
        if start < 0 or end <= start or end > len(audio):
            raise ValueError("Wake phrase interval is outside the positive recording")
        if end - start > 16000:
            raise ValueError("Wake phrase must fit inside the one-second model context")

    if len(audio) <= 16000:
        return [fit_window(audio, 16000)]
    if start is None or end is None:
        center = max(0, (len(audio) - 16000) // 2)
        return [audio[center:center + 16000].copy()]
    min_start = max(0, end - 16000)
    max_start = min(start, len(audio) - 16000)
    if min_start > max_start:
        raise ValueError("No one-second crop can contain the complete wake phrase")
    count = max(1, min(5, int(limit)))
    if min_start == max_start:
        starts = np.asarray([min_start], dtype=int)
    else:
        starts = np.unique(rng.integers(min_start, max_start + 1, size=count))
        if len(starts) < count:
            starts = np.unique(np.r_[starts, np.linspace(min_start, max_start, count, dtype=int)])
    return [audio[int(offset):int(offset) + 16000].copy() for offset in starts]


def _features(clip: np.ndarray) -> np.ndarray:
    feat = log_mel(fit_window(clip, 16000))
    if feat.shape != (98, 40):
        raise RuntimeError(f"Frontend produced unexpected shape {feat.shape}; expected (98, 40)")
    return feat


def prepare(project: str | Path, seed: int | None = None, config_path: str | Path | None = None,
            augment: bool | None = None) -> dict:
    project = Path(project)
    config = load_config(config_path or (project / "wakeword.yaml" if (project / "wakeword.yaml").exists() else None))
    seed = int(config["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    rows = scan(project)
    labelled = [r for r in rows if r["label"] is not None]
    labelled_groups = {r["source_group"] for r in labelled}
    # Background-only recordings are augmentation assets, not labelled examples.
    # Default noise-only groups to train so they cannot consume validation/test
    # split slots or accidentally become a hidden source of validation leakage.
    # Noise recordings sharing a source_group with labelled recordings inherit
    # that group's split through the normal group-aware splitter.
    for row in rows:
        if (row["kind"] == "noise" and row["source_group"] not in labelled_groups
                and row["split"] is None):
            row["split"] = "train"
    if not labelled:
        raise ValueError("No positive/negative WAV files found under data/raw/{positive,negative}")
    if not any(r["label"] == 1 for r in labelled) or not any(r["label"] == 0 for r in labelled):
        raise ValueError("Both positive and negative recordings are required")

    fractions = tuple(float(config["split"][key]) for key in ("train", "validation", "test"))
    splits = group_split(rows, seed=seed, fractions=fractions)
    # Keep only short snippets from train-partition ambient audio for mixing; never
    # hold full multi-hour noise recordings (or validation/test noise) in memory.
    noise_snippets: list[np.ndarray] = []
    for row in splits["train"]:
        if row["kind"] != "noise":
            continue
        ambient = read_wav(row["path"])
        if len(ambient) >= 16000:
            for _ in range(min(3, max(1, len(ambient) // 16000))):
                start = int(rng.integers(0, len(ambient) - 16000 + 1))
                noise_snippets.append(ambient[start:start + 16000].copy())
        elif len(ambient):
            noise_snippets.append(fit_window(ambient, 16000))
        del ambient

    do_augment = bool(config["augmentation"]["enabled"] if augment is None else augment)
    aug_cfg = config["augmentation"]
    ds_cfg = config["dataset"]
    mix_cfg = config["noise_mixing"]
    do_noise_mix = bool(mix_cfg["enabled"])
    copies_per_clip = int(mix_cfg["copies_per_clip"])
    mix_snr_min = float(mix_cfg["snr_min_db"])
    mix_snr_max = float(mix_cfg["snr_max_db"])
    out = project / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split_name, items in splits.items():
        xs: list[np.ndarray] = []
        ys: list[int] = []
        positives = [r for r in items if r["kind"] == "positive"]
        negatives = [r for r in items if r["kind"] == "negative"]
        noise = noise_snippets if split_name == "train" else []
        # Noise-mixing sources for this partition only. Split integrity contract:
        # a window may only ever be mixed with noise from a noise group that
        # belongs to the SAME partition (fetch-noise split pins route them).
        noise_rows = [r for r in items if r["kind"] == "noise"]
        noise_cache: dict[str, np.ndarray | None] = {}
        mixed_count = 0
        if do_noise_mix and not noise_rows:
            warnings.warn(
                f"noise_mixing enabled but split {split_name!r} has no noise groups; "
                "skipping mixing for this partition")

        def _partition_noise_snippet() -> np.ndarray | None:
            if not noise_rows:
                return None
            row = noise_rows[int(rng.integers(0, len(noise_rows)))]
            if row["path"] not in noise_cache:
                ambient = read_wav(row["path"])
                noise_cache[row["path"]] = ambient if len(ambient) else None
            ambient = noise_cache[row["path"]]
            if ambient is None:
                return None
            if len(ambient) >= 16000:
                start = int(rng.integers(0, len(ambient) - 16000 + 1))
                return ambient[start:start + 16000]
            return fit_window(ambient, 16000)

        def _append_mixed_copies(clip: np.ndarray, label: int) -> None:
            nonlocal mixed_count
            if not do_noise_mix:
                return
            for _ in range(copies_per_clip):
                snippet = _partition_noise_snippet()
                if snippet is None:
                    continue
                snr = float(rng.uniform(mix_snr_min, mix_snr_max))
                xs.append(_features(mix_with_noise(clip, snippet, snr)))
                ys.append(label)
                mixed_count += 1

        if split_name == "train" and ds_cfg.get("noise_as_negative_train", True):
            # Background-only recordings are both hard negative examples and
            # augmentation sources, but are never copied into validation/test.
            for ambient_clip in noise_snippets:
                xs.append(_features(ambient_clip))
                ys.append(0)

        for row in positives:
            audio = read_wav(row["path"])
            clips = _positive_clips(
                audio, rng, int(ds_cfg["max_windows_per_recording"]),
                row.get("phrase_start_s"), row.get("phrase_end_s"),
            )
            del audio
            for clip in clips:
                xs.append(_features(clip)); ys.append(1)
                if split_name == "train" and do_augment:
                    for _ in range(int(aug_cfg["copies_per_positive"])):
                        generated = augment_audio(
                            clip, rng, noise, sample_rate=16000,
                            speed_min=float(aug_cfg["speed_min"]), speed_max=float(aug_cfg["speed_max"]),
                            pitch_semitones=float(aug_cfg["pitch_semitones"]),
                            max_shift_ms=int(aug_cfg["max_shift_ms"]),
                            reverb_probability=float(aug_cfg["reverb_probability"]),
                            noise_probability=float(aug_cfg["noise_probability"]),
                            snr_min_db=float(aug_cfg["snr_min_db"]), snr_max_db=float(aug_cfg["snr_max_db"]),
                        )
                        xs.append(_features(generated)); ys.append(1)
                _append_mixed_copies(clip, 1)

        negative_hop = round(16000 * float(ds_cfg["negative_hop_ms"]) / 1000)
        for row in negatives:
            audio = read_wav(row["path"])
            max_windows = int(ds_cfg["max_windows_per_recording"])
            if len(audio) <= 16000:
                starts = [0]
            else:
                total_windows = 1 + (len(audio) - 16000) // negative_hop
                indices = (np.linspace(0, total_windows - 1, max_windows, dtype=int)
                           if total_windows > max_windows else np.arange(total_windows))
                starts = indices * negative_hop
            for start in starts:
                clip = fit_window(audio[int(start):int(start) + 16000], 16000)
                xs.append(_features(clip)); ys.append(0)
                _append_mixed_copies(clip, 0)
            del audio

        if not xs or len(set(ys)) < 2:
            raise ValueError(f"Split {split_name!r} has no examples of one class; add independent groups or adjust split pins")
        np.savez_compressed(out / f"{split_name}.npz", x=np.asarray(xs, np.float32), y=np.asarray(ys, np.int64))
        summary[split_name] = {
            "examples": len(ys), "positive": int(sum(ys)), "negative": int(len(ys) - sum(ys)),
            "noise_mixed": mixed_count,
        }

    manifest = {
        "seed": seed,
        "splits": {name: [{"path": row["path"], "source_group": row["source_group"], "kind": row["kind"], "split": name} for row in items] for name, items in splits.items()},
        "summary": summary,
        "augmentation": {"enabled": do_augment, "train_only": True},
        "noise_mixing": {"enabled": do_noise_mix},
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
