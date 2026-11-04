from __future__ import annotations
from pathlib import Path
import json
import warnings
import numpy as np

from .config import load_config
from .metrics import choose_threshold, classification_metrics


def _normalizer(artifact_dir: Path):
    values = json.loads((artifact_dir / "normalization.json").read_text(encoding="utf-8"))
    mean, std = np.asarray(values["mean"], np.float32), np.asarray(values["std"], np.float32)
    if mean.shape != (40,) or std.shape != (40,) or np.any(std <= 0):
        raise ValueError("Bad normalization metadata: expected 40 means and positive standard deviations")
    return mean, std


def _balanced_indices(labels: np.ndarray, maximum: int, seed: int):
    rng = np.random.default_rng(seed)
    per_class = max(1, maximum // 2)
    selected = []
    for cls in (0, 1):
        candidates = np.flatnonzero(labels == cls)
        if not len(candidates):
            raise ValueError("INT8 calibration requires examples from both classes")
        selected.extend(rng.choice(candidates, size=min(len(candidates), per_class), replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)


def export(project, run_name="default", config_path=None):
    import tensorflow as tf
    project = Path(project)
    artifact_dir = project / "artifacts" / "experiments" / run_name
    config = load_config(config_path or (project / "wakeword.yaml" if (project / "wakeword.yaml").exists() else None))
    model = tf.keras.models.load_model(artifact_dir / "model.keras")
    mean, std = _normalizer(artifact_dir)
    with np.load(project / "data/processed/validation.npz") as archive:
        x_val, y_val = archive["x"].copy(), archive["y"].copy()
    with np.load(project / "data/processed/train.npz") as archive:
        x_train, y_train = archive["x"].copy(), archive["y"].copy()
    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Validation data must contain both classes for threshold calibration")
    x_val_norm = ((x_val - mean) / std)[..., None].astype(np.float32)

    float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    float_model = float_converter.convert()
    (artifact_dir / "wakeword_float32.tflite").write_bytes(float_model)

    calib_n = int(config["quantization"]["calibration_samples"])
    calib_idx = _balanced_indices(y_train, calib_n, int(config["seed"]))
    x_calib = ((x_train[calib_idx] - mean) / std)[..., None].astype(np.float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([sample[None, ...]] for sample in x_calib)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    quant_model = converter.convert()
    quant_path = artifact_dir / "wakeword_int8.tflite"
    quant_path.write_bytes(quant_model)

    interpreter = tf.lite.Interpreter(model_content=quant_model, num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise RuntimeError("Exporter did not produce fully INT8 input and output tensors")
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    if in_scale <= 0 or out_scale <= 0:
        raise RuntimeError("Quantized model reported invalid affine scale")

    quant_scores = []
    float_scores = model.predict(x_val_norm, batch_size=128, verbose=0).reshape(-1)
    for sample in x_val_norm:
        q = np.clip(np.rint(sample / in_scale + in_zero), -128, 127).astype(np.int8)[None, ...]
        interpreter.set_tensor(input_detail["index"], q)
        interpreter.invoke()
        raw = interpreter.get_tensor(output_detail["index"]).astype(np.float32).reshape(-1)[0]
        quant_scores.append(float((raw - out_zero) * out_scale))
    quant_scores = np.clip(np.asarray(quant_scores, np.float32), 0.0, 1.0)

    target_fpr = float(config["quantization"]["target_validation_fpr"])
    threshold = choose_threshold(quant_scores, y_val, target_fpr)
    neg_count = int(np.sum(y_val == 0))
    achievable_fpr_step = 1.0 / max(1, neg_count)
    if achievable_fpr_step > target_fpr:
        warnings.warn(
            f"Only {neg_count} validation negatives: empirical FPR resolution is {achievable_fpr_step:.4g}, "
            f"so target {target_fpr:.4g} cannot be established with this validation set.", RuntimeWarning,
        )

    meta = {
        "schema_version": 1,
        "model_file": "wakeword_int8.tflite",
        "sample_rate": 16000, "window_samples": 16000,
        "feature_shape": [98, 40], "input_shape": [int(dim) for dim in input_detail["shape"]],
        "output_shape": [int(dim) for dim in output_detail["shape"]],
        "frame_length_ms": 25, "frame_step_ms": 10, "fft_length": 512,
        "mel_bins": 40, "lower_hz": 60.0, "upper_hz": 7600.0,
        "window": "symmetric_hann", "mel_scale": "htk",
        "mean": mean.tolist(), "std": std.tolist(),
        "threshold": threshold, "threshold_target_validation_fpr": target_fpr,
        "validation_negative_windows": neg_count,
        "input_quantization": {"scale": float(in_scale), "zero_point": int(in_zero), "dtype": "int8"},
        "output_quantization": {"scale": float(out_scale), "zero_point": int(out_zero), "dtype": "int8"},
        "validation_quantized_metrics": classification_metrics(quant_scores, y_val, threshold),
        "quantization_score_mae": float(np.mean(np.abs(float_scores - quant_scores))),
        "quantization_score_max_abs_error": float(np.max(np.abs(float_scores - quant_scores))),
        "calibration_samples": int(len(calib_idx)),
        "model_size_bytes": int(quant_path.stat().st_size),
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
