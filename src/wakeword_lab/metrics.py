from __future__ import annotations
import numpy as np


def choose_threshold(scores, labels, target_fpr=0.001):
    """Choose the lowest threshold whose empirical validation FPR is <= target_fpr."""
    # Use float32 because the exact threshold is serialized to JSON and parsed as a
    # Kotlin Float on Android. A float64-only nextafter can round back onto the max
    # negative score and silently reintroduce false accepts at the boundary.
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    y = np.asarray(labels).astype(bool).reshape(-1)
    if s.shape != y.shape or not len(s):
        raise ValueError("scores and labels must be non-empty arrays of equal length")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0,1)")
    neg = s[~y]
    if not len(neg):
        raise ValueError("validation requires negative examples")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    candidates = np.unique(neg)
    above_max = np.nextafter(np.float32(np.max(neg)), np.float32(np.inf))
    candidates = np.r_[candidates, above_max]
    valid = [float(t) for t in candidates if np.mean(neg >= t) <= target_fpr]
    return min(valid)


def classification_metrics(scores, labels, threshold):
    s = np.asarray(scores).reshape(-1)
    y = np.asarray(labels).astype(bool).reshape(-1)
    if s.shape != y.shape or not len(s):
        raise ValueError("scores and labels must be non-empty arrays of equal length")
    if not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    pred = s >= threshold
    tp = int(np.sum(pred & y)); tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    return {
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
    }


def false_activations_per_hour(event_times, duration_seconds, merge_gap=1.0):
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if not np.isfinite(merge_gap) or merge_gap < 0:
        raise ValueError("merge_gap must be finite and non-negative")
    times = sorted(float(t) for t in event_times)
    if any(not np.isfinite(t) or t < 0 or t > duration_seconds for t in times):
        raise ValueError("event times must be finite and within the recording duration")
    count, last = 0, -float("inf")
    for t in times:
        if t - last >= merge_gap:
            count += 1
        last = t
    return count * 3600.0 / duration_seconds
