from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .metrics import classification_metrics


def evaluate_test(project, run_name="default"):
    import tensorflow as tf
    project = Path(project)
    art = project / "artifacts/experiments" / run_name
    meta = json.loads((art / "model_metadata.json").read_text(encoding="utf-8"))
    with np.load(project / "data/processed/test.npz") as archive:
        x, y = archive["x"].copy(), archive["y"].copy()
    mean, std = np.asarray(meta["mean"], np.float32), np.asarray(meta["std"], np.float32)
    x = ((x - mean) / std)[..., None].astype(np.float32)
    model_bytes = (art / meta["model_file"]).read_bytes()
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale, zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]
    scores = []
    for sample in x:
        q = np.clip(np.rint(sample / scale + zero), -128, 127).astype(np.int8)[None, ...]
        interpreter.set_tensor(inp["index"], q); interpreter.invoke()
        value = interpreter.get_tensor(out["index"]).astype(np.float32).reshape(-1)[0]
        scores.append(float(np.clip((value - out_zero) * out_scale, 0.0, 1.0)))
    metrics = classification_metrics(scores, y, float(meta["threshold"]))
    metrics["threshold"] = float(meta["threshold"])
    metrics["test_examples"] = int(len(y))
    metrics["model_size_bytes"] = int((art / meta["model_file"]).stat().st_size)
    (art / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
