"""Offline tests for per-run report generation (no TFLite models are loaded)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _fabricate_run(project, run_name="demo", seed=8, threshold=0.42):
    art = project / "artifacts/experiments" / run_name
    art.mkdir(parents=True)
    header = "epoch,accuracy,loss,precision,recall,val_accuracy,val_loss,val_precision,val_recall"
    rows = [
        "1,0.60,0.65,0.58,0.62,0.61,0.63,0.59,0.63",
        "2,0.75,0.45,0.73,0.70,0.74,0.44,0.72,0.71",
        "3,0.82,0.34,0.81,0.79,0.80,0.36,0.78,0.77",
    ]
    (art / "training.csv").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    (art / "model_metadata.json").write_text(json.dumps({
        "threshold": threshold,
        "threshold_target_validation_fpr": 0.001,
        "mean": [0.0] * 40,
        "std": [1.0] * 40,
        "model_size_bytes": 4096,
        "quantization_score_mae": 0.01,
        "quantization_score_max_abs_error": 0.05,
        "validation_quantized_metrics": {
            "accuracy": 0.9, "precision": 0.88, "recall": 0.86,
            "false_positives": 3, "false_negatives": 4,
        },
    }), encoding="utf-8")
    (art / "run_config.json").write_text(json.dumps({
        "run_name": run_name, "seed": seed, "width": 8, "dropout": 0.1,
        "learning_rate": 0.001, "epochs_requested": 10, "epochs_completed": 3,
    }), encoding="utf-8")
    (art / "summary.json").write_text(json.dumps({
        "accuracy": 0.85, "precision": 0.84, "recall": 0.82,
        "false_positives": 5, "false_negatives": 6, "threshold": threshold,
        "test_examples": 20,
    }), encoding="utf-8")
    processed = project / "data/processed"
    processed.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for split, count in (("validation", 16), ("test", 12)):
        x = rng.normal(0, 1, (count, 98, 40)).astype(np.float32)
        y = np.array([0, 1] * (count // 2), dtype=np.int64)
        np.savez_compressed(processed / f"{split}.npz", x=x, y=y)
    manifest = b"{}"
    (processed / "split_manifest.json").write_bytes(manifest)
    digest = hashlib.sha256(manifest).hexdigest()[:12]
    return art, digest


@pytest.fixture()
def patched_scoring(monkeypatch):
    """Replace TFLite/Keras scoring seams with deterministic synthetic score arrays."""
    from wakeword_lab import reporting

    def fake_int8(project, run_name, split):
        rng = np.random.default_rng(abs(hash((run_name, split))) & 0xFFFF)
        _, y = reporting._load_split(Path(project), split)
        scores = np.where(y == 1, rng.normal(0.8, 0.08, len(y)), rng.normal(0.3, 0.1, len(y)))
        return np.clip(scores, 0.0, 1.0).astype(np.float32), y

    def fake_float32(project, run_name, split):
        scores, y = fake_int8(project, run_name, split)
        return np.clip(scores + np.random.default_rng(1).normal(0, 0.005, len(scores)), 0.0, 1.0), y

    monkeypatch.setattr(reporting, "_score_int8_split", fake_int8)
    monkeypatch.setattr(reporting, "_score_float32_split", fake_float32)
    return reporting


def test_generate_report_creates_all_artifacts(tmp_path, monkeypatch, patched_scoring):
    pytest.importorskip("matplotlib")
    art, digest = _fabricate_run(tmp_path)
    report_dir = patched_scoring.generate_report(tmp_path, "demo")
    assert report_dir == art / "report"
    for name in ("training_curves.png", "score_distributions.png", "roc_pr.png",
                 "quantization_drift.png", "report.md"):
        path = report_dir / name
        assert path.is_file() and path.stat().st_size > 0, f"{name} missing or empty"
    text = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "demo" in text                      # run name
    assert "Seed: 8" in text                   # seed from run_config.json
    assert digest in text                      # split_manifest.json sha256 prefix
    assert "0.42" in text                      # calibrated threshold


def test_roc_curve_perfect_separation_has_auc_one():
    from wakeword_lab.reporting import roc_curve
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    labels = np.array([1, 1, 1, 0, 0, 0])
    fpr, tpr, auc = roc_curve(scores, labels)
    assert auc == pytest.approx(1.0)
    assert fpr[0] == 0.0 and tpr[0] == 0.0


def test_roc_curve_rejects_mismatched_inputs():
    from wakeword_lab.reporting import roc_curve
    with pytest.raises(ValueError):
        roc_curve(np.array([0.1, 0.2]), np.array([1]))


def test_precision_recall_vs_threshold_marks_endpoint():
    from wakeword_lab.reporting import precision_recall_vs_threshold
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    thresholds, precision, recall = precision_recall_vs_threshold(scores, labels)
    assert thresholds[0] == 0.9 and precision[0] == 1.0 and recall[0] == 0.5
    assert recall[-1] == 1.0


def test_missing_matplotlib_raises_clear_runtimeerror(tmp_path, monkeypatch, patched_scoring):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # force ImportError on import
    _fabricate_run(tmp_path)
    with pytest.raises(RuntimeError, match="train extra"):
        patched_scoring.generate_report(tmp_path, "demo")


def test_run_experiment_survives_report_failure(tmp_path, monkeypatch):
    from wakeword_lab import experiment

    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/train.npz").touch()
    (tmp_path / "artifacts/experiments/demo").mkdir(parents=True)
    monkeypatch.setattr(experiment, "train", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "export", lambda *args, **kwargs: {})
    monkeypatch.setattr(experiment, "evaluate_test", lambda *args, **kwargs: {"accuracy": 0.9})
    monkeypatch.setattr(experiment, "generate_report",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    metrics = experiment.run_experiment(tmp_path, "demo")
    assert metrics["accuracy"] == 0.9  # report failure did not fail the experiment


def test_run_experiment_calls_generate_report(tmp_path, monkeypatch):
    from wakeword_lab import experiment

    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/train.npz").touch()
    (tmp_path / "artifacts/experiments/demo").mkdir(parents=True)
    monkeypatch.setattr(experiment, "train", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, "export", lambda *args, **kwargs: {})
    monkeypatch.setattr(experiment, "evaluate_test", lambda *args, **kwargs: {"accuracy": 0.9})
    calls = []
    monkeypatch.setattr(experiment, "generate_report", lambda *args: calls.append(args))
    experiment.run_experiment(tmp_path, "demo")
    assert calls == [(tmp_path, "demo")]
