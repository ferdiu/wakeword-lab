"""Interactive microphone recording into the raw data folders.

Requires the optional record extra (sounddevice + soundfile). Capture runs at
the given sample rate as mono int16; each accepted take is published as a
pcm16 WAV under data/raw/<kind>/ and tracked in metadata.csv.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np

from .synthesize import _append_metadata

MAX_TAKE_SECONDS = 10.0
MIN_TAKE_SECONDS = 0.3
RECORD_EXTRA_ERROR = "Install the record extra with `pip install -e '.[record]'` before recording"


def _require_soundfile() -> None:
    if importlib.util.find_spec("soundfile") is None:
        raise RuntimeError(RECORD_EXTRA_ERROR)


def _safe_speaker(speaker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", speaker).strip("_") or "unknown"


def record_take(sample_rate: int = 16000, max_seconds: float = MAX_TAKE_SECONDS) -> np.ndarray:
    """Capture one take from the default microphone as mono int16 PCM.

    Starts capture immediately and returns once the user presses Enter or the
    take reaches ``max_seconds`` (auto-stop with a notice). Returns an empty
    array if no audio arrived.
    """
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(RECORD_EXTRA_ERROR) from error
    blocks: list[np.ndarray] = []
    state = {"frames": 0}
    stop = threading.Event()

    def _callback(indata, frame_count, time_info, status):
        if status:
            print(f"Warning: audio input status {status}; frames may have been dropped.")
        blocks.append(indata.copy())
        state["frames"] += frame_count

    def _wait_for_enter():
        try:
            input("Recording... press Enter to stop.")
        except EOFError:
            pass  # stdin closed (e.g. piped input): treat it as a stop request
        stop.set()

    max_frames = int(round(max_seconds * sample_rate))
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=1600, callback=_callback):
        waiter = threading.Thread(target=_wait_for_enter, daemon=True)
        waiter.start()
        hit_limit = False
        while not stop.wait(timeout=0.05):
            if state["frames"] >= max_frames:
                hit_limit = True
                break
        if hit_limit:
            print(f"Reached the {max_seconds:.0f}s limit; stopping this take.")
    if not blocks:
        return np.zeros(0, dtype=np.int16)
    pcm = np.concatenate([block.reshape(-1) for block in blocks])[:max_frames]
    return np.ascontiguousarray(pcm, dtype=np.int16)


def _prompt_count(input_fn) -> int:
    """Ask for the take count, tolerating one non-numeric answer."""
    for _attempt in range(2):
        raw = input_fn("How many samples to record? [20]: ").strip()
        if not raw:
            return 20
        try:
            return int(raw)
        except ValueError:
            print(f"'{raw}' is not a whole number.")
    raise SystemExit("Invalid sample count entered twice; aborting the session.")


def run_recording_session(project, *, kind: str = "positive", speaker: str = "unknown",
                          split: str | None = None, sample_rate: int = 16000,
                          count: int | None = None,
                          capture=None, input_fn=input) -> int:
    """Record ``count`` takes interactively and publish them under data/raw/<kind>/.

    ``capture`` and ``input_fn`` are injectable so tests can substitute fakes.
    Takes shorter than MIN_TAKE_SECONDS are rejected and do not consume the
    count, so the user can redo them. Sessions are deterministic given the
    same takes.
    """
    if kind not in ("positive", "negative", "noise"):
        raise ValueError(f"kind must be positive, negative or noise, got {kind!r}")
    if split not in (None, "train", "validation", "test"):
        raise ValueError(f"split must be train, validation, test or blank, got {split!r}")
    # sounddevice is only needed when using the real capture path (record_take
    # checks it itself); soundfile is always needed to publish the WAV.
    _require_soundfile()
    if capture is None:
        capture = record_take
    if count is None:
        count = _prompt_count(input_fn)
    if count < 1:
        raise ValueError("count must be at least one")
    project = Path(project)
    out_dir = project / "data" / "raw" / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = project / "data" / "raw" / "metadata.csv"
    run_id = uuid.uuid4().hex[:8]
    saved = 0
    while saved < count:
        input_fn(f"Take {saved + 1}/{count} — press Enter to start recording...")
        pcm = np.asarray(capture(sample_rate, MAX_TAKE_SECONDS))
        duration = len(pcm) / float(sample_rate)
        if duration < MIN_TAKE_SECONDS:
            print(f"Take too short ({duration:.2f}s, need at least {MIN_TAKE_SECONDS}s); redo it.")
            continue
        wav = out_dir / f"{_safe_speaker(speaker)}__rec_{run_id}_{saved:03d}.wav"
        temporary = wav.with_suffix(".tmp.wav")  # soundfile infers the format from the extension
        import soundfile as sf
        sf.write(temporary, pcm, sample_rate, subtype="PCM_16")
        shutil.move(temporary, wav)
        try:
            _append_metadata(metadata_path, [{
                "path": wav.name, "speaker_id": speaker, "group_id": f"{speaker}_rec_{run_id}",
                "split": split or "", "locale": "", "phrase_start_s": "", "phrase_end_s": "",
            }])
        except Exception:
            wav.unlink(missing_ok=True)
            raise
        saved += 1
    print(f"Saved {saved} recordings to {out_dir}")
    return saved
