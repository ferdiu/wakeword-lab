import csv
import re

import numpy as np
import pytest

from wakeword_lab.cli import _project_init
from wakeword_lab.record import run_recording_session


def _project(tmp_path):
    project = tmp_path / "proj"
    _project_init(project, "Hey Nova")
    return project


def _scripted_input(lines):
    queue = list(lines)

    def _input(prompt=""):
        print(prompt, end="")
        return queue.pop(0) if queue else ""

    return _input


def _metadata_rows(project):
    with (project / "data/raw/metadata.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_session_saves_named_wavs_and_metadata_rows(tmp_path):
    project = _project(tmp_path)
    takes = [np.full(16000, 100, dtype=np.int16) for _ in range(2)]
    capture_calls = []

    def capture(sample_rate, max_seconds):
        capture_calls.append((sample_rate, max_seconds))
        return takes.pop(0)

    saved = run_recording_session(
        project, kind="positive", speaker="Alice Smith", count=2,
        capture=capture, input_fn=_scripted_input(["", ""]))

    assert saved == 2
    assert capture_calls == [(16000, 10.0), (16000, 10.0)]
    wavs = sorted((project / "data/raw/positive").glob("*.wav"))
    assert len(wavs) == 2
    names = [wav.name for wav in wavs]
    matches = [re.match(r"^Alice_Smith__rec_([0-9a-f]{8})_(\d{3})\.wav$", name) for name in names]
    assert all(matches)
    run_ids = {match.group(1) for match in matches}
    assert len(run_ids) == 1
    assert [match.group(2) for match in matches] == ["000", "001"]

    rows = _metadata_rows(project)
    assert len(rows) == 2
    run_id = run_ids.pop()
    for row in rows:
        assert row["speaker_id"] == "Alice Smith"
        assert row["group_id"] == f"Alice Smith_rec_{run_id}"
        assert row["split"] == ""
        assert row["locale"] == ""
        assert row["phrase_start_s"] == "" and row["phrase_end_s"] == ""
    # WAV content is real pcm16 audio at 16 kHz.
    import soundfile as sf
    audio, rate = sf.read(wavs[0], dtype="int16")
    assert rate == 16000 and audio.ndim == 1 and len(audio) == 16000


def test_short_take_is_rejected_without_consuming_count(tmp_path):
    project = _project(tmp_path)
    durations = [0.1, 1.0, 1.0]  # first take too short, then two good ones

    def capture(sample_rate, max_seconds):
        return np.zeros(int(durations.pop(0) * sample_rate), dtype=np.int16)

    saved = run_recording_session(
        project, speaker="bob", count=2, capture=capture,
        input_fn=_scripted_input(["", "", ""]))
    assert saved == 2
    wavs = sorted((project / "data/raw/positive").glob("*.wav"))
    # Successful takes keep contiguous indices despite the rejected first take.
    stems = [wav.name[:-8] for wav in wavs]  # strip "_NNN.wav"
    assert len(set(stems)) == 1
    assert [wav.name for wav in wavs] == [f"{stems[0]}_000.wav", f"{stems[0]}_001.wav"]
    assert len(_metadata_rows(project)) == 2


def test_split_is_pinned_when_given(tmp_path):
    project = _project(tmp_path)
    run_recording_session(
        project, speaker="carol", split="validation", count=1,
        capture=lambda rate, max_s: np.zeros(rate, dtype=np.int16),
        input_fn=_scripted_input([""]))
    rows = _metadata_rows(project)
    assert len(rows) == 1
    assert rows[0]["split"] == "validation"
    assert rows[0]["group_id"].endswith("_rec_" + rows[0]["path"].split("_rec_")[1][:8])


def test_empty_prompt_uses_default_count_of_twenty(tmp_path):
    project = _project(tmp_path)
    remaining = {"n": 20}

    def capture(sample_rate, max_seconds):
        remaining["n"] -= 1
        return np.zeros(sample_rate, dtype=np.int16)

    saved = run_recording_session(
        project, speaker="dave", count=None, capture=capture,
        input_fn=_scripted_input([""]))
    assert saved == 20
    assert remaining["n"] == 0
    assert len(list((project / "data/raw/positive").glob("*.wav"))) == 20
    assert len(_metadata_rows(project)) == 20


def test_invalid_count_prompt_recovers_once(tmp_path):
    project = _project(tmp_path)
    saved = run_recording_session(
        project, speaker="frank", count=None,
        capture=lambda rate, max_s: np.zeros(rate, dtype=np.int16),
        input_fn=_scripted_input(["lots", "1", ""]))
    assert saved == 1


def test_invalid_count_prompt_twice_exits_friendly(tmp_path, capsys):
    project = _project(tmp_path)
    with pytest.raises(SystemExit):
        run_recording_session(
            project, speaker="gina", count=None,
            capture=lambda rate, max_s: np.zeros(rate, dtype=np.int16),
            input_fn=_scripted_input(["lots", "more"]))
    out = capsys.readouterr().out
    assert "not a whole number" in out


def test_negative_kind_lands_in_negative_folder(tmp_path):
    project = _project(tmp_path)
    run_recording_session(
        project, kind="negative", speaker="erin", count=1,
        capture=lambda rate, max_s: np.zeros(rate, dtype=np.int16),
        input_fn=_scripted_input([""]))
    assert len(list((project / "data/raw/negative").glob("erin__rec_*.wav"))) == 1
    assert list((project / "data/raw/positive").glob("*.wav")) == []
