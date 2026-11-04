import csv
from wakeword_lab import experiment

def test_sweep_does_not_touch_test_set(tmp_path, monkeypatch):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/processed/train.npz").touch()
    trained = []
    monkeypatch.setattr(experiment, "train", lambda project, **kwargs: trained.append(kwargs["run_name"]))
    monkeypatch.setattr(experiment, "export", lambda project, run_name, config_path=None: {
        "validation_quantized_metrics": {
            "accuracy": .9, "precision": .8, "recall": .7,
            "false_positives": 2, "false_negatives": 3,
        },
        "model_size_bytes": 4096, "quantization_score_mae": .01,
    })
    monkeypatch.setattr(experiment, "evaluate_test", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("sweep must not read final test set")
    ))
    rows = experiment.sweep(tmp_path, [8, 12], [.1])
    assert len(rows) == 2 and len(trained) == 2
    with (tmp_path / "artifacts/sweep_results.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["validation_recall"] == "0.7"
    assert "test_accuracy" not in saved[0]
