import asyncio
import csv
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from wakeword_lab import synthesize
from wakeword_lab.cli import _project_init


@pytest.fixture
def project(tmp_path):
    target = tmp_path / "proj"
    _project_init(target, "Hey Nova")
    return target


@pytest.fixture
def fake_tts(monkeypatch):
    """Replace the edge_tts/ffmpeg layer with deterministic offline fakes."""
    module = types.ModuleType("edge_tts")
    voices = [
        {"Locale": "en-IN", "ShortName": "en-IN-VoiceA"},
        {"Locale": "en-US", "ShortName": "en-US-VoiceA"},
        {"Locale": "en-GB", "ShortName": "en-GB-VoiceA"},
        {"Locale": "en-AU", "ShortName": "en-AU-VoiceA"},
    ]

    async def list_voices():
        return voices

    class Communicate:
        def __init__(self, text, voice, rate=None, pitch=None):
            self.text = text

        async def save(self, path):
            Path(path).write_bytes(b"fake mp3 payload")

    module.list_voices = list_voices
    module.Communicate = Communicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)

    def fake_run(cmd, check=True, timeout=None):
        output = Path(cmd[-1])
        pcm = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(np.round(pcm * 32767).astype(np.int16).tobytes())

    monkeypatch.setattr(synthesize.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(synthesize.subprocess, "run", fake_run)
    return module


def _metadata_rows(project):
    with (project / "data/raw/metadata.csv").open(newline="", encoding="utf-8") as stream:
        return {row["path"]: row for row in csv.DictReader(stream)}


def test_synthesize_negatives_uses_negative_prefix_and_no_phrase_timestamps(project, fake_tts):
    count = asyncio.run(synthesize.synthesize_negatives(project, ["open the pod bay doors"], 1, 7))
    assert count == 6
    wavs = sorted((project / "data/raw/negative").glob("*.wav"))
    assert len(wavs) == 6
    assert all(wav.name.startswith("neg_tts_") for wav in wavs)
    assert list((project / "data/raw/positive").glob("*.wav")) == []

    rows = _metadata_rows(project)
    assert set(rows) == {wav.name for wav in wavs}
    for row in rows.values():
        assert row["split"] == "train"
        assert row["locale"].startswith("en-")
        assert row["phrase_start_s"] == "" and row["phrase_end_s"] == ""


def test_synthesize_negatives_multiple_phrases_scale_utterances(project, fake_tts):
    count = asyncio.run(synthesize.synthesize_negatives(project, ["hello there", "not now"], 2, 3))
    assert count == 24
    assert len(list((project / "data/raw/negative").glob("neg_tts_*.wav"))) == 24


def test_synthesize_negatives_rejects_empty_phrases(project, fake_tts):
    with pytest.raises(ValueError, match="phrase cannot be empty"):
        asyncio.run(synthesize.synthesize_negatives(project, ["  "], 1, 3))


def test_synthesize_positives_keep_phrase_timestamps(project, fake_tts):
    count = asyncio.run(synthesize.synthesize(project, "Hey Nova", 1, 5))
    assert count == 6
    wavs = sorted((project / "data/raw/positive").glob("tts_*.wav"))
    assert len(wavs) == 6
    rows = _metadata_rows(project)
    for wav in wavs:
        row = rows[wav.name]
        assert row["split"] == "train"
        assert 0.0 <= float(row["phrase_start_s"]) < float(row["phrase_end_s"]) <= 1.0


def test_read_phrases_file_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "phrases.txt"
    path.write_text("# hard negatives\nhello there\n\n  not the wake phrase  \n", encoding="utf-8")
    assert synthesize._read_phrases_file(path) == ["hello there", "not the wake phrase"]
