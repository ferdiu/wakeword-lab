import csv
from wakeword_lab.synthesize import _append_metadata

def test_metadata_append_preserves_custom_fields_and_split_pins(tmp_path):
    path = tmp_path / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "speaker_id", "split", "locale", "take_note"])
        writer.writeheader()
        writer.writerow({"path": "human.wav", "speaker_id": "me", "split": "validation", "locale": "en-GB", "take_note": "held-out"})
    _append_metadata(path, [{"path": "tts_1.wav", "speaker_id": "en-IN-voice", "split": "train", "locale": "en-IN"}])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["path"]: row for row in csv.DictReader(stream)}
    assert rows["human.wav"]["take_note"] == "held-out"
    assert rows["human.wav"]["split"] == "validation"
    assert rows["tts_1.wav"]["split"] == "train"
    assert rows["tts_1.wav"]["locale"] == "en-IN"


def test_speech_interval_finds_complete_generated_phrase():
    import numpy as np
    from wakeword_lab.synthesize import _speech_interval

    sample_rate = 16000
    audio = np.zeros(sample_rate, np.float32)
    t = np.arange(2400, 11200) / sample_rate
    audio[2400:11200] = 0.1 * np.sin(2 * np.pi * 220 * t)
    start, end = _speech_interval(audio)
    assert start <= 0.15 and end >= 0.70
    assert end - start <= 1.0


def test_speech_interval_rejects_phrase_longer_than_model_context():
    import numpy as np
    import pytest
    from wakeword_lab.synthesize import _speech_interval

    with pytest.raises(ValueError, match="shorter wake phrase"):
        _speech_interval(np.ones(24000, np.float32) * 0.1)


def test_speech_interval_keeps_long_but_valid_phrase_with_reduced_padding():
    import numpy as np
    from wakeword_lab.synthesize import _speech_interval

    audio = np.zeros(16000, np.float32)
    t = np.arange(800, 15200) / 16000
    audio[800:15200] = 0.1 * np.sin(2 * np.pi * 180 * t)
    start, end = _speech_interval(audio)
    assert end - start <= 1.0
    assert start <= 0.05 and end >= 0.95
