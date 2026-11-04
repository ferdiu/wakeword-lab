from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .metrics import false_activations_per_hour


def _intervals(annotation):
    result = []
    for item in annotation or []:
        if isinstance(item, dict):
            start, end = float(item["start"]), float(item["end"])
        else:
            start, end = map(float, item)
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid annotation interval: {item}")
        result.append((start, end))
    return sorted(result)




def _iter_audio_windows(blocks, sample_rate=16000, window_samples=16000, hop_samples=320):
    """Yield (one-second chronological window, 20 ms chunk, timestamp) from mono/stereo blocks.

    Blocks may end at arbitrary sample boundaries. Only a fixed ring buffer and less than
    one hop of carry are retained, so long recordings are not loaded into memory at once.
    """
    ring = np.zeros(window_samples, dtype=np.float32)
    write_index = 0
    ring_count = 0
    samples_seen = 0
    carry = np.empty(0, dtype=np.float32)
    for block in blocks:
        data = np.asarray(block, dtype=np.float32)
        mono = data.mean(axis=1) if data.ndim == 2 else data
        if mono.ndim != 1:
            raise ValueError("Audio blocks must be mono or [frames, channels]")
        combined = np.concatenate((carry, mono)) if len(carry) else mono
        complete = (len(combined) // hop_samples) * hop_samples
        for offset in range(0, complete, hop_samples):
            chunk = combined[offset:offset + hop_samples]
            first = min(hop_samples, window_samples - write_index)
            ring[write_index:write_index + first] = chunk[:first]
            if first < hop_samples:
                ring[:hop_samples - first] = chunk[first:]
            write_index = (write_index + hop_samples) % window_samples
            ring_count = min(window_samples, ring_count + hop_samples)
            samples_seen += hop_samples
            if ring_count == window_samples:
                window = np.concatenate((ring[write_index:], ring[:write_index])) if write_index else ring.copy()
                yield window, chunk, samples_seen / sample_rate
        carry = combined[complete:].copy()


def evaluate_stream(project, recordings_dir, run_name="default", annotations_path=None,
                    vad_rms=0.008, cooldown_ms=1500, merge_gap=1.0, threshold_override=None):
    """Replay long audio like the Android detector and report false triggers/hour + interval recall."""
    import tensorflow as tf
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("Install the training extra (pip install -e '.[train]') for streaming WAV evaluation") from error
    project = Path(project)
    art = project / "artifacts/experiments" / run_name
    meta = json.loads((art / "model_metadata.json").read_text(encoding="utf-8"))
    mean = np.asarray(meta["mean"], np.float32); std = np.asarray(meta["std"], np.float32)
    threshold = float(meta["threshold"] if threshold_override is None else threshold_override)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    if not np.isfinite(vad_rms) or vad_rms < 0 or cooldown_ms < 0 or not np.isfinite(merge_gap) or merge_gap < 0:
        raise ValueError("VAD threshold, cooldown and merge gap must be finite and non-negative")
    interpreter = tf.lite.Interpreter(model_content=(art / meta["model_file"]).read_bytes(), num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]; out_scale, out_zero = out["quantization"]
    annotations = {}
    if annotations_path:
        annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))

    recordings_root = Path(recordings_dir)
    recordings = sorted(recordings_root.rglob("*.wav"))
    if not recordings:
        raise ValueError(f"No WAV recordings found in {recordings_dir}")
    results = {}
    all_false_events = 0
    total_seconds = 0.0
    total_intervals = 0
    detected_intervals = 0
    for wav in recordings:
        relative_name = wav.relative_to(recordings_root).as_posix()
        with sf.SoundFile(wav) as audio_file:
            if audio_file.samplerate != 16000:
                raise ValueError(
                    f"Continuous-evaluation WAVs must be 16 kHz to match Android capture; "
                    f"{relative_name} is {audio_file.samplerate} Hz. Convert it first with "
                    f"ffmpeg -i input.wav -ar 16000 -ac 1 output.wav."
                )
            duration = audio_file.frames / audio_file.samplerate
            if duration < 1.0:
                continue
            truth = _intervals(annotations.get(relative_name, annotations.get(wav.name, annotations.get(str(wav), []))))
            if any(end > duration for _, end in truth):
                raise ValueError(f"Annotation interval exceeds recording duration for {relative_name}")
            events = []
            last_fire = -float("inf")
            from .frontend import log_mel
            batch_features, batch_times = [], []

            def flush_batch():
                nonlocal last_fire
                if not batch_features:
                    return
                normalized = ((np.stack(batch_features) - mean) / std)[..., None].astype(np.float32)
                for sample, timestamp in zip(normalized, batch_times):
                    q = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8)[None, ...]
                    interpreter.set_tensor(inp["index"], q)
                    interpreter.invoke()
                    raw = interpreter.get_tensor(out["index"]).astype(np.float32).reshape(-1)[0]
                    score = float(np.clip((raw - out_zero) * out_scale, 0.0, 1.0))
                    if score >= threshold and (timestamp - last_fire) * 1000 >= cooldown_ms:
                        events.append(timestamp)
                        last_fire = timestamp
                batch_features.clear(); batch_times.clear()

            # SoundFile reads at most ten seconds at a time; _iter_audio_windows keeps
            # only a one-second ring and 20 ms hop carry between blocks.
            blocks = audio_file.blocks(blocksize=16000 * 10, dtype="float32", always_2d=True)
            for window, chunk, timestamp in _iter_audio_windows(blocks):
                rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                if rms >= vad_rms:
                    batch_features.append(log_mel(window))
                    batch_times.append(timestamp)
                    if len(batch_features) >= 32:
                        flush_batch()
            flush_batch()

        matched = set()
        false_events = []
        for event in events:
            match = next((i for i, (start, end) in enumerate(truth)
                          if i not in matched and start - 0.5 <= event <= end + 0.5), None)
            if match is None:
                false_events.append(event)
            else:
                matched.add(match)
        total_intervals += len(truth)
        detected_intervals += len(matched)
        false_bursts = int(round(false_activations_per_hour(false_events, duration, merge_gap) * duration / 3600.0))
        all_false_events += false_bursts
        total_seconds += duration
        results[relative_name] = {
            "duration_seconds": duration, "events": events,
            "false_events": false_events, "false_event_bursts": false_bursts,
            "annotated_intervals": len(truth),
            "detected_intervals": len(matched), "missed_intervals": len(truth) - len(matched),
        }
    if total_seconds <= 0:
        raise ValueError("No recording at least one second long was evaluated")
    report = {
        "recordings": results, "duration_hours": total_seconds / 3600.0,
        "false_activations": all_false_events,
        "false_activations_per_hour": all_false_events * 3600.0 / total_seconds,
        "annotated_wake_words": total_intervals,
        "detected_wake_words": detected_intervals,
        "wake_word_recall": detected_intervals / total_intervals if total_intervals else None,
        "vad_rms": float(vad_rms), "cooldown_ms": int(cooldown_ms),
        "threshold": threshold,
        "threshold_source": "model_metadata" if threshold_override is None else "cli_override",
        "note": "For meaningful FA/hour confidence, use multiple hours of truly held-out ambient recordings.",
    }
    (art / "continuous_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
