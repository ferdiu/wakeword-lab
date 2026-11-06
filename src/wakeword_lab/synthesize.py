"""Optional Edge TTS generator. Requires network access, edge-tts and ffmpeg.
Generated speech is stored as 16 kHz mono WAV and recorded in metadata.csv.
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import os
import random
import shutil
import uuid
import subprocess
import tempfile
from pathlib import Path
import numpy as np


def _speech_interval(audio: np.ndarray, sample_rate: int = 16000, frame_samples: int = 160,
                    relative_threshold: float = 0.025, padding_seconds: float = 0.06) -> tuple[float, float]:
    """Estimate the non-silent speech span in generated TTS audio.

    This is only a coarse energy VAD for synthetic clips, not a general forced aligner.
    It lets the dataset crop the complete generated phrase rather than blindly center-crop
    a long TTS file, and rejects phrases that cannot fit the one-second model context.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples) or sample_rate <= 0 or frame_samples <= 0:
        raise ValueError("Expected non-empty mono/stereo TTS audio and a positive sample rate")
    frame_count = (len(samples) + frame_samples - 1) // frame_samples
    rms = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        frame = samples[index * frame_samples:(index + 1) * frame_samples]
        rms[index] = np.sqrt(np.mean(frame * frame) + 1e-12)
    peak = float(rms.max(initial=0.0))
    threshold = max(0.001, peak * relative_threshold)
    active = np.flatnonzero(rms >= threshold)
    if not len(active):
        raise ValueError("TTS output contains no detectable speech energy")
    active_start = active[0] * frame_samples / sample_rate
    active_end = min(len(samples) / sample_rate, (active[-1] + 1) * frame_samples / sample_rate)
    speech_duration = active_end - active_start
    if speech_duration > 1.0:
        raise ValueError(
            f"Generated phrase spans {speech_duration:.2f}s and cannot fit the one-second model context; "
            "use a shorter wake phrase"
        )
    # Keep as much leading/trailing context as fits, but do not reject a valid phrase
    # merely because symmetric padding would push its interval past one second.
    padding = min(padding_seconds, max(0.0, (1.0 - speech_duration) / 2.0))
    start = max(0.0, active_start - padding)
    end = min(len(samples) / sample_rate, active_end + padding)
    return round(start, 4), round(end, 4)


def _append_metadata(path: Path, rows: list[dict]) -> None:
    existing = []
    fields = ["path", "speaker_id", "split", "locale"]
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            existing = list(reader)
            fields = list(dict.fromkeys(fields + (reader.fieldnames or [])))
    by_path = {row.get("path"): row for row in existing}
    for row in rows:
        merged = dict(by_path.get(row["path"], {}))
        merged.update(row)
        by_path[row["path"]] = merged
        fields = list(dict.fromkeys(fields + list(merged.keys())))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in by_path.values():
                writer.writerow({key: row.get(key, "") for key in fields})
            stream.flush()
            os.fsync(stream.fileno())
        shutil.move(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def synthesize(project, phrase, variants_per_locale=24, seed=42):
    try:
        import edge_tts
    except ImportError as error:
        raise RuntimeError("Install the TTS extra with `pip install -e '.[tts]'` before synthesis") from error
    if variants_per_locale < 1:
        raise ValueError("variants_per_locale must be at least one")
    if not phrase.strip():
        raise ValueError("phrase cannot be empty")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg executable not found; install ffmpeg to convert Edge TTS audio to 16 kHz WAV")
    project = Path(project)
    out = project / "data/raw/positive"
    out.mkdir(parents=True, exist_ok=True)
    voices = await asyncio.wait_for(edge_tts.list_voices(), timeout=45)
    pools = {
        "en-IN": [v for v in voices if v["Locale"].startswith("en-IN")],
        "en-US": [v for v in voices if v["Locale"].startswith("en-US")],
        "en-GB": [v for v in voices if v["Locale"].startswith("en-GB")],
        "en-AU": [v for v in voices if v["Locale"].startswith("en-AU")],
    }
    if any(not v for v in pools.values()):
        raise RuntimeError("Edge TTS did not return one or more requested English voice locales")
    rng = random.Random(seed)
    # Article's 3x en-IN weighting is represented as weighted locale selection per utterance.
    locale_weights = [("en-IN", 3), ("en-US", 1), ("en-GB", 1), ("en-AU", 1)]
    metadata_path = project / "data/raw/metadata.csv"
    run_id = uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix="wakeword-tts-") as temp_dir:
        temp_dir = Path(temp_dir)
        for index in range(int(variants_per_locale) * 6):
            locale = rng.choices([x[0] for x in locale_weights], weights=[x[1] for x in locale_weights])[0]
            voice = rng.choice(pools[locale])
            rate = rng.randint(-15, 15)
            pitch = rng.randint(-3, 3)
            speaker = voice["ShortName"]
            safe_speaker = speaker.replace("-", "_")
            base = f"tts_{run_id}_{index:05d}_{safe_speaker}"
            mp3 = temp_dir / f"{base}.mp3"
            wav_tmp = temp_dir / f"{base}.wav"
            wav = out / f"{base}.wav"
            await asyncio.wait_for(
                edge_tts.Communicate(
                    phrase, speaker, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz"
                ).save(str(mp3)),
                timeout=45,
            )
            subprocess.run([
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(mp3), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_tmp),
            ], check=True, timeout=60)
            if not wav_tmp.is_file() or wav_tmp.stat().st_size <= 44:
                raise RuntimeError(f"ffmpeg produced an empty or invalid WAV for {base}")
            import soundfile as sf
            generated_audio, generated_rate = sf.read(wav_tmp, dtype="float32", always_2d=False)
            if generated_rate != 16000:
                raise RuntimeError(f"ffmpeg produced {generated_rate} Hz instead of 16000 Hz for {base}")
            phrase_start, phrase_end = _speech_interval(generated_audio, generated_rate)
            # Atomic publish: an interrupted conversion cannot leave an untracked positive WAV
            # that a later dataset scan might incorrectly assign to validation or test.
            shutil.move(wav_tmp, wav)
            try:
                _append_metadata(metadata_path, [{
                    "path": wav.name, "speaker_id": speaker, "split": "train", "locale": locale,
                    "phrase_start_s": phrase_start, "phrase_end_s": phrase_end,
                }])
            except Exception:
                wav.unlink(missing_ok=True)
                raise
    return int(variants_per_locale) * 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--variants-per-locale", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(asyncio.run(synthesize(args.project, args.phrase, args.variants_per_locale, args.seed)))


if __name__ == "__main__":
    main()
