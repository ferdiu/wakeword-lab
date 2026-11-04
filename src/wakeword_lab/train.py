from __future__ import annotations

from pathlib import Path
import json
import random
import numpy as np

from .config import load_config


def train(project, epochs=None, batch_size=None, run_name="default", config_path=None,
          width=None, dropout=None, learning_rate=None):
    """Train one experiment. `run_name` creates an isolated artifacts/experiments/<name> folder."""
    import tensorflow as tf
    from .model import build_model

    project = Path(project)
    config = load_config(config_path or (project / "wakeword.yaml" if (project / "wakeword.yaml").exists() else None))
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

    data_dir = project / "data/processed"
    with np.load(data_dir / "train.npz") as archive:
        x_train, y_train = archive["x"].copy(), archive["y"].copy()
    with np.load(data_dir / "validation.npz") as archive:
        x_val, y_val = archive["x"].copy(), archive["y"].copy()
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Training and validation datasets must each contain positive and negative samples")

    mean = x_train.mean(axis=(0, 1)).astype(np.float32)
    std = (x_train.std(axis=(0, 1)) + 1e-5).astype(np.float32)
    normalize = lambda x: ((x - mean) / std)[..., None]
    model_cfg = config["model"]
    selected_width = int(model_cfg["width"] if width is None else width)
    selected_dropout = float(model_cfg["dropout"] if dropout is None else dropout)
    selected_learning_rate = float(model_cfg["learning_rate"] if learning_rate is None else learning_rate)
    selected_epochs = int(model_cfg["epochs"] if epochs is None else epochs)
    selected_batch_size = int(model_cfg["batch_size"] if batch_size is None else batch_size)
    if selected_width < 4 or selected_epochs < 1 or selected_batch_size < 1:
        raise ValueError("width must be >=4 and epochs/batch_size must be positive")
    if not np.isfinite(selected_dropout) or not 0 <= selected_dropout < 1:
        raise ValueError("dropout must be finite and in [0,1)")
    if not np.isfinite(selected_learning_rate) or selected_learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    model = build_model(
        width=selected_width,
        dropout=selected_dropout,
        label_smoothing=float(model_cfg["label_smoothing"]),
        learning_rate=selected_learning_rate,
    )
    artifact_dir = project / "artifacts" / "experiments" / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    class_weight = {i: float(len(y_train) / (2 * max(1, class_counts[i]))) for i in (0, 1)}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(model_cfg["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(artifact_dir / "best.keras"), monitor="val_loss", save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(str(artifact_dir / "training.csv")),
    ]
    history = model.fit(
        normalize(x_train), y_train,
        validation_data=(normalize(x_val), y_val),
        epochs=selected_epochs,
        batch_size=selected_batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    model.save(artifact_dir / "model.keras")
    (artifact_dir / "normalization.json").write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2), encoding="utf-8"
    )
    (artifact_dir / "run_config.json").write_text(json.dumps({
        "run_name": run_name, "seed": seed, "width": selected_width,
        "dropout": selected_dropout,
        "learning_rate": selected_learning_rate,
        "epochs_requested": selected_epochs,
        "epochs_completed": len(history.history.get("loss", [])),
        "class_weight": class_weight,
    }, indent=2), encoding="utf-8")
    return model
