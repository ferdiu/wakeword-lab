"""Per-run report generation: plots and a markdown summary under artifacts/experiments/<run>/report/.

Matplotlib is an optional dependency (the `train` extra); everything here fails with a
clear RuntimeError when it is missing. INT8 scoring reuses evaluation.score_int8_windows
so quantization/dequantization logic is never duplicated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import score_int8_windows

FIGSIZE = (8, 5)
DPI = 110
SCORE_SPLITS = ("validation", "test")  # train is skipped: re-scoring it adds no calibration insight


def _load_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Install the train extra (pip install 'wakeword-lab[train]') to generate run reports: "
            "matplotlib is required for plotting."
        ) from exc
    matplotlib.use("Agg")  # headless backend; never require a display
    from matplotlib import pyplot as plt
    return plt


def roc_curve(scores, labels):
    """Pure-numpy ROC: returns (fpr, tpr, auc) with the origin prepended."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(bool).reshape(-1)
    if s.shape != y.shape or not len(s) or not np.isfinite(s).all():
        raise ValueError("scores and labels must be non-empty finite arrays of equal length")
    order = np.argsort(-s, kind="stable")
    positives = max(1, int(np.sum(y)))
    negatives = max(1, int(np.sum(~y)))
    tps = np.cumsum(y[order])
    fps = np.cumsum(~y[order])
    tpr = np.r_[0.0, tps / positives]
    fpr = np.r_[0.0, fps / negatives]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def precision_recall_vs_threshold(scores, labels):
    """Pure-numpy precision/recall evaluated at every distinct score threshold (descending)."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).astype(bool).reshape(-1)
    if s.shape != y.shape or not len(s) or not np.isfinite(s).all():
        raise ValueError("scores and labels must be non-empty finite arrays of equal length")
    thresholds = np.unique(s)[::-1]
    pred = s[None, :] >= thresholds[:, None]
    tp = np.sum(pred & y[None, :], axis=1)
    fp = np.sum(pred & ~y[None, :], axis=1)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(np.sum(y)))
    return thresholds, precision, recall


def _load_meta(project: Path, run_name: str) -> dict:
    meta = json.loads((project / "artifacts/experiments" / run_name / "model_metadata.json")
                      .read_text(encoding="utf-8"))
    return meta


def _load_split(project: Path, split: str):
    with np.load(project / "data/processed" / f"{split}.npz") as archive:
        return archive["x"].copy(), archive["y"].copy()


def _normalize(x: np.ndarray, meta: dict) -> np.ndarray:
    mean = np.asarray(meta["mean"], np.float32)
    std = np.asarray(meta["std"], np.float32)
    return ((x - mean) / std)[..., None].astype(np.float32)


def _score_int8_split(project, run_name, split):
    """Load a processed split, normalize with the run's stats and score with the INT8 model."""
    project = Path(project)
    meta = _load_meta(project, run_name)
    x, y = _load_split(project, split)
    scores = score_int8_windows(project, run_name, _normalize(x, meta))
    return scores, y


def _score_float32_split(project, run_name, split):
    """Score a processed split with the run's saved float32 Keras model."""
    import tensorflow as tf
    project = Path(project)
    art = project / "artifacts/experiments" / run_name
    meta = _load_meta(project, run_name)
    x, y = _load_split(project, split)
    model = tf.keras.models.load_model(art / "model.keras")
    scores = model.predict(_normalize(x, meta), batch_size=128, verbose=0).reshape(-1)
    return np.asarray(scores, np.float32), y


def _manifest_digest(project: Path) -> str:
    path = project / "data/processed/split_manifest.json"
    if not path.exists():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _plot_training_curves(art: Path, report_dir: Path, plt):
    import csv
    rows = list(csv.DictReader((art / "training.csv").open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"{art / 'training.csv'} has no epoch rows")
    epoch = np.arange(1, len(rows) + 1)
    fig, (ax_loss, ax_metrics) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
    ax_loss.plot(epoch, [float(r["loss"]) for r in rows], label="train")
    ax_loss.plot(epoch, [float(r["val_loss"]) for r in rows], label="validation")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    for key, label in (("accuracy", "accuracy"), ("precision", "precision"), ("recall", "recall")):
        ax_metrics.plot(epoch, [float(r[key]) for r in rows], label=f"train {label}")
        ax_metrics.plot(epoch, [float(r[f"val_{key}"]) for r in rows], label=f"val {label}", linestyle="--")
    last_epoch = len(rows)
    for ax in (ax_loss, ax_metrics):
        ax.axvline(last_epoch, color="red", linestyle=":", alpha=0.7, label="early-stopping end")
    ax_metrics.set_xlabel("epoch")
    ax_metrics.set_ylabel("metric")
    ax_metrics.legend(fontsize="small")
    ax_metrics.grid(alpha=0.3)
    ax_loss.set_title("training curves")
    fig.tight_layout()
    fig.savefig(report_dir / "training_curves.png", dpi=DPI)
    plt.close(fig)
    return len(rows)


def _plot_score_distributions(report_dir: Path, plt, split_scores, threshold: float):
    fig, axes = plt.subplots(len(split_scores), 1, figsize=FIGSIZE)
    if len(split_scores) == 1:
        axes = [axes]
    for ax, (split, scores, labels) in zip(axes, split_scores):
        pos, neg = scores[labels == 1], scores[labels == 0]
        ax.hist(neg, bins=40, range=(0.0, 1.0), alpha=0.6, label=f"negative (n={len(neg)})")
        ax.hist(pos, bins=40, range=(0.0, 1.0), alpha=0.6, label=f"positive (n={len(pos)})")
        ax.axvline(threshold, color="red", linestyle="--", label=f"threshold {threshold:.4g}")
        ax.set_title(f"{split} INT8 score distribution")
        ax.set_xlabel("score")
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(report_dir / "score_distributions.png", dpi=DPI)
    plt.close(fig)


def _plot_roc_pr(report_dir: Path, plt, scores, labels, threshold: float):
    fpr, tpr, auc = roc_curve(scores, labels)
    thresholds, precision, recall = precision_recall_vs_threshold(scores, labels)
    pred_pos = scores >= threshold
    tpr_at = float(np.mean(np.asarray(labels)[pred_pos])) if np.any(pred_pos) else 0.0
    neg = ~np.asarray(labels).astype(bool)
    fpr_at = float(np.mean(np.asarray(scores)[neg] >= threshold)) if np.any(neg) else 0.0
    prec_at = tpr_at * np.sum(labels) / max(1, np.sum(pred_pos))

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=FIGSIZE)
    ax_roc.plot(fpr, tpr, label=f"ROC (AUC {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="gray", label="chance")
    ax_roc.plot(fpr_at, tpr_at, "ro", label=f"threshold {threshold:.4g}")
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("recall (TPR)")
    ax_roc.set_title("validation ROC (INT8)")
    ax_roc.legend(fontsize="small")
    ax_roc.grid(alpha=0.3)
    ax_pr.plot(thresholds, precision, label="precision")
    ax_pr.plot(thresholds, recall, label="recall")
    ax_pr.axvline(threshold, color="red", linestyle="--", label=f"threshold {threshold:.4g} (P {prec_at:.3f})")
    ax_pr.set_xlabel("threshold")
    ax_pr.set_ylabel("value")
    ax_pr.set_title("validation precision/recall vs threshold")
    ax_pr.legend(fontsize="small")
    ax_pr.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "roc_pr.png", dpi=DPI)
    plt.close(fig)


def _plot_quantization_drift(report_dir: Path, plt, float_scores, int8_scores, meta):
    mae_meta = float(meta.get("quantization_score_mae", float("nan")))
    max_abs_meta = float(meta.get("quantization_score_max_abs_error", float("nan")))
    mae_recomputed = float(np.mean(np.abs(float_scores - int8_scores)))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(float_scores, int8_scores, s=8, alpha=0.4, label="validation windows")
    lims = [0.0, 1.0]
    ax.plot(lims, lims, linestyle=":", color="gray", label="y = x")
    ax.set_xlabel("float32 score")
    ax.set_ylabel("int8 score")
    ax.set_title("quantization drift (validation, paired scores)")
    ax.annotate(
        f"MAE recomputed {mae_recomputed:.5f} (metadata {mae_meta:.5f})\n"
        f"max |err| metadata {max_abs_meta:.5f}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize="small",
    )
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(report_dir / "quantization_drift.png", dpi=DPI)
    plt.close(fig)
    return mae_recomputed


def _metrics_table_row(split: str, metrics: dict, threshold: float) -> str:
    def fmt(value):
        return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)

    return (
        f"| {split} | {fmt(metrics.get('accuracy', float('nan')))} "
        f"| {fmt(metrics.get('precision', float('nan')))} | {fmt(metrics.get('recall', float('nan')))} "
        f"| {metrics.get('false_positives', '?')} | {metrics.get('false_negatives', '?')} "
        f"| {threshold:.6g} |"
    )


def _build_markdown(project: Path, run_name: str, art: Path, meta: dict, epochs_completed: int,
                    mae_recomputed: float) -> str:
    run_config = json.loads((art / "run_config.json").read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])
    summary_path = art / "summary.json"
    test_metrics = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists()
                    else json.loads((art / "test_metrics.json").read_text(encoding="utf-8")))
    lines = [
        f"# Run report: {run_name}",
        "",
        f"- Seed: {run_config.get('seed')}",
        f"- Data manifest sha256 (first 12 hex): `{_manifest_digest(project)}`",
        f"- Epochs completed: {epochs_completed} (requested {run_config.get('epochs_requested')})",
        f"- Calibrated threshold: {threshold:.6g} "
        f"(target validation FPR {meta.get('threshold_target_validation_fpr')})",
        f"- Model size: {meta.get('model_size_bytes')} bytes",
        f"- Quantization score MAE: {meta.get('quantization_score_mae'):.6g} "
        f"(recomputed on validation drift plot: {mae_recomputed:.6g})",
        "",
        "## Run configuration",
        "",
        "```json",
        json.dumps(run_config, indent=2),
        "```",
        "",
        "## Metrics at calibrated threshold",
        "",
        "| Split | Accuracy | Precision | Recall | FP | FN | Threshold |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        _metrics_table_row("validation", meta.get("validation_quantized_metrics", {}), threshold),
        _metrics_table_row("test", test_metrics, threshold),
        "",
        "## Files",
        "",
        "- `training_curves.png` — train vs validation loss and metrics per epoch",
        "- `score_distributions.png` — INT8 score histograms (positive vs negative) per split",
        "- `roc_pr.png` — validation ROC and precision/recall vs threshold",
        "- `quantization_drift.png` — paired float32 vs INT8 validation scores",
        "",
    ]
    return "\n".join(lines)


def generate_report(project, run_name="default") -> Path:
    """Generate (or refresh) artifacts/experiments/<run>/report/; returns the report directory."""
    project = Path(project)
    art = project / "artifacts/experiments" / run_name
    meta = _load_meta(project, run_name)
    threshold = float(meta["threshold"])
    report_dir = art / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    plt = _load_matplotlib()
    epochs_completed = _plot_training_curves(art, report_dir, plt)

    split_scores = []
    val_scores = val_labels = None
    for split in SCORE_SPLITS:
        scores, labels = _score_int8_split(project, run_name, split)
        split_scores.append((split, scores, labels))
        if split == "validation":
            val_scores, val_labels = scores, labels
    _plot_score_distributions(report_dir, plt, split_scores, threshold)
    _plot_roc_pr(report_dir, plt, val_scores, val_labels, threshold)

    float_scores, _ = _score_float32_split(project, run_name, "validation")
    mae_recomputed = _plot_quantization_drift(report_dir, plt, float_scores, val_scores, meta)

    (report_dir / "report.md").write_text(
        _build_markdown(project, run_name, art, meta, epochs_completed, mae_recomputed),
        encoding="utf-8",
    )
    return report_dir
