import pytest
from wakeword_lab.config import DEFAULT_CONFIG, validate_config

def test_default_config_validates():
    validate_config(DEFAULT_CONFIG)

def test_android_contract_rejects_wrong_sample_rate():
    cfg = {**DEFAULT_CONFIG, "sample_rate": 8000}
    with pytest.raises(ValueError): validate_config(cfg)


def test_dataset_window_cap_must_be_positive():
    cfg = {**DEFAULT_CONFIG, "dataset": {**DEFAULT_CONFIG["dataset"], "max_windows_per_recording": 0}}
    with pytest.raises(ValueError):
        validate_config(cfg)

def test_quantization_requires_both_classes():
    cfg = {**DEFAULT_CONFIG, "quantization": {**DEFAULT_CONFIG["quantization"], "calibration_samples": 1}}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_frontend_mel_range_is_locked_to_android_contract():
    cfg = {**DEFAULT_CONFIG, "upper_hz": 7000.0}
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_binary_label_smoothing_maps_to_article_confidence_targets():
    from wakeword_lab.model import keras_binary_smoothing
    import numpy as np

    keras_amount = keras_binary_smoothing(0.05)
    # Keras BCE target transform: y * (1 - amount) + 0.5 * amount.
    labels = np.array([0.0, 1.0])
    targets = labels * (1.0 - keras_amount) + 0.5 * keras_amount
    assert np.allclose(targets, [0.05, 0.95])


def test_binary_label_smoothing_rejects_invalid_values():
    import pytest
    from wakeword_lab.model import keras_binary_smoothing

    with pytest.raises(ValueError):
        keras_binary_smoothing(0.5)


def test_boolean_dataset_config_must_be_boolean():
    cfg = {**DEFAULT_CONFIG, "dataset": {**DEFAULT_CONFIG["dataset"], "noise_as_negative_train": "false"}}
    with pytest.raises(ValueError):
        validate_config(cfg)
