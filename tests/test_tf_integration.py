"""Optional end-to-end training + full INT8 conversion smoke test (requires .[train])."""
import json
import numpy as np
import pytest


def test_train_export_and_test_evaluation(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from wakeword_lab.train import train
    from wakeword_lab.export import export
    from wakeword_lab.evaluation import evaluate_test

    project = tmp_path
    data = project / "data/processed"
    data.mkdir(parents=True)
    rng = np.random.default_rng(8)
    for split, count in (("train", 12), ("validation", 8), ("test", 8)):
        x = rng.normal(0, 1, (count, 98, 40)).astype(np.float32)
        y = np.array([0, 1] * (count // 2), dtype=np.int64)
        x[y == 1, 45:55, 10:20] += 0.4
        np.savez_compressed(data / f"{split}.npz", x=x, y=y)
    config = {
        "seed": 8, "sample_rate": 16000, "clip_seconds": 1.0,
        "frame_length_ms": 25, "frame_step_ms": 10, "fft_length": 512,
        "mel_bins": 40, "lower_hz": 60.0, "upper_hz": 7600.0,
        "split": {"train": .7, "validation": .15, "test": .15},
        "augmentation": {"enabled": False, "copies_per_positive": 0, "speed_min": .8,
                         "speed_max": 1.2, "pitch_semitones": 4, "max_shift_ms": 200,
                         "reverb_probability": .3, "noise_probability": .5,
                         "snr_min_db": 5, "snr_max_db": 25},
        "dataset": {"negative_hop_ms": 500, "max_windows_per_recording": 80},
        "model": {"width": 8, "dropout": .1, "epochs": 1, "batch_size": 4,
                  "learning_rate": .001, "label_smoothing": .05, "early_stopping_patience": 1},
        "quantization": {"calibration_samples": 8, "target_validation_fpr": .5},
    }
    config_path = project / "test.yaml"
    config_path.write_text(json.dumps(config))
    train(project, run_name="smoke", config_path=config_path, epochs=1, batch_size=4, width=8)
    metadata = export(project, run_name="smoke", config_path=config_path)
    metrics = evaluate_test(project, run_name="smoke")
    assert (project / "artifacts/experiments/smoke/wakeword_int8.tflite").is_file()
    assert metadata["input_quantization"]["dtype"] == "int8"
    assert metadata["output_quantization"]["dtype"] == "int8"
    assert 0 <= metrics["accuracy"] <= 1
