from __future__ import annotations
from pathlib import Path
import csv
import itertools
import json

from .dataset import prepare
from .train import train
from .export import export
from .evaluation import evaluate_test
from .reporting import generate_report


def run_experiment(project, run_name="default", config_path=None, **overrides):
    project = Path(project)
    if not (project / "data/processed/train.npz").exists():
        prepare(project, config_path=config_path)
    train(project, run_name=run_name, config_path=config_path, **overrides)
    export(project, run_name=run_name, config_path=config_path)
    metrics = evaluate_test(project, run_name=run_name)
    (project / "artifacts/experiments" / run_name / "summary.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    try:
        generate_report(project, run_name)
    except Exception as exc:  # a failed report must never fail the experiment itself
        print(f"warning: report generation failed for run {run_name!r}: {exc}")
    return metrics


def sweep(project, widths, dropouts, config_path=None):
    """Train/export candidate models and rank by validation only; never inspect test here."""
    project = Path(project)
    if not (project / "data/processed/train.npz").exists():
        prepare(project, config_path=config_path)
    results = []
    for width, dropout in itertools.product(widths, dropouts):
        name = f"w{int(width)}_d{str(dropout).replace('.', 'p')}"
        train(project, run_name=name, config_path=config_path, width=int(width), dropout=float(dropout))
        metadata = export(project, run_name=name, config_path=config_path)
        validation = metadata.get("validation_quantized_metrics", {})
        results.append({
            "run_name": name,
            "width": int(width),
            "dropout": float(dropout),
            "validation_accuracy": validation.get("accuracy"),
            "validation_precision": validation.get("precision"),
            "validation_recall": validation.get("recall"),
            "validation_false_positives": validation.get("false_positives"),
            "validation_false_negatives": validation.get("false_negatives"),
            "model_size_bytes": metadata.get("model_size_bytes"),
            "quantization_score_mae": metadata.get("quantization_score_mae"),
        })
    path = project / "artifacts/sweep_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]) if results else ["run_name"])
        writer.writeheader()
        writer.writerows(results)
    return results
