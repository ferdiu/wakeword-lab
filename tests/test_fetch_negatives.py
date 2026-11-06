import io
import zipfile

import pytest

from wakeword_lab import fetch_negatives
from wakeword_lab.cli import _project_init


def _zip_bytes(entries):
    """entries: list of (archive name, payload bytes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _mini_archive(words=("up", "down"), per_word=4):
    entries = []
    for word in words:
        for index in range(per_word):
            entries.append((f"{word}/clip_{index:04d}.wav", b"RIFF-fake-wav"))
    entries.append((f"{words[0]}/README.txt", b"not audio"))
    return _zip_bytes(entries)


def _project(tmp_path):
    project = tmp_path / "proj"
    _project_init(project, "Hey Nova")
    return project


def _fake_downloader(payload):
    def _download(url, destination):
        with open(destination, "wb") as stream:
            stream.write(payload)

    return _download


def test_fetch_selects_balanced_clips_with_word_prefix_names(tmp_path, capsys):
    project = _project(tmp_path)
    payload = _mini_archive(words=("up", "down"), per_word=4)
    paths = fetch_negatives.fetch_negatives(
        project, count=4, seed=1, downloader=_fake_downloader(payload))

    assert len(paths) == 4
    names = sorted(path.name for path in paths)
    assert all(name.startswith("sc_") and "__" in name for name in names)
    per_word = {"up": 0, "down": 0}
    for name in names:
        word = name[len("sc_"):].split("__")[0]
        per_word[word] += 1
    assert per_word == {"up": 2, "down": 2}  # uniform across word folders
    # Only .wav members are extracted.
    assert "sc_up__README.txt" not in names
    summary = capsys.readouterr().out
    assert "up: 2" in summary and "down: 2" in summary


def test_fetch_is_seed_reproducible(tmp_path):
    project_a, project_b = _project(tmp_path / "a"), _project(tmp_path / "b")
    payload = _mini_archive(per_word=6)
    args = dict(count=4, seed=9, downloader=_fake_downloader(payload))
    names_a = sorted(p.name for p in fetch_negatives.fetch_negatives(project_a, **args))
    names_b = sorted(p.name for p in fetch_negatives.fetch_negatives(project_b, **args))
    assert names_a == names_b and names_a


def test_fetch_rejects_path_traversal_members(tmp_path):
    project = _project(tmp_path)
    payload = _zip_bytes([
        ("up/clip_0000.wav", b"RIFF-fake-wav"),
        ("../evil.wav", b"RIFF-fake-wav"),
    ])
    with pytest.raises(ValueError, match="unsafe archive member"):
        fetch_negatives.fetch_negatives(
            project, count=2, seed=1, downloader=_fake_downloader(payload))
    assert list((project / "data/raw/negative").glob("*.wav")) == []


def test_fetch_rejects_archive_without_wavs(tmp_path):
    project = _project(tmp_path)
    payload = _zip_bytes([("up/README.txt", b"no audio here")])
    with pytest.raises(RuntimeError, match="no .wav entries"):
        fetch_negatives.fetch_negatives(
            project, count=2, seed=1, downloader=_fake_downloader(payload))
