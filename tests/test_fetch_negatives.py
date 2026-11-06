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
    """Mirror the real archive layout: clips nest under a top-level
    mini_speech_commands/ directory, alongside a non-WAV member."""
    entries = []
    for word in words:
        for index in range(per_word):
            entries.append((f"mini_speech_commands/{word}/clip_{index:04d}.wav",
                            b"RIFF-fake-wav"))
    entries.append(("mini_speech_commands/README.md", b"not audio"))
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
    # The top-level mini_speech_commands/ directory must not become the word:
    # a fetch across words yields one sc_<word>__ prefix per real word folder.
    prefixes = {name[len("sc_"):].split("__")[0] for name in names}
    assert prefixes == {"up", "down"}
    # Only .wav members are extracted.
    assert not any("README" in name for name in names)
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
        ("mini_speech_commands/up/clip_0000.wav", b"RIFF-fake-wav"),
        ("../evil.wav", b"RIFF-fake-wav"),
    ])
    with pytest.raises(ValueError, match="unsafe archive member"):
        fetch_negatives.fetch_negatives(
            project, count=2, seed=1, downloader=_fake_downloader(payload))
    assert list((project / "data/raw/negative").glob("*.wav")) == []


def test_fetch_rejects_archive_without_wavs(tmp_path):
    project = _project(tmp_path)
    payload = _zip_bytes([("mini_speech_commands/README.md", b"no audio here")])
    with pytest.raises(RuntimeError, match="no .wav entries"):
        fetch_negatives.fetch_negatives(
            project, count=2, seed=1, downloader=_fake_downloader(payload))


def test_fetch_skips_macosx_appledouble_members(tmp_path):
    project = _project(tmp_path)
    entries = [(f"mini_speech_commands/up/clip_{index:04d}.wav", b"RIFF-fake-wav")
               for index in range(3)]
    # The real archive ships AppleDouble metadata files for every clip.
    entries += [(f"__MACOSX/mini_speech_commands/up/._clip_{index:04d}.wav", b"junk")
                for index in range(3)]
    paths = fetch_negatives.fetch_negatives(
        project, count=3, seed=1, downloader=_fake_downloader(_zip_bytes(entries)))
    assert len(paths) == 3
    assert all(not path.name.startswith("sc_up__._") for path in paths)


def test_fetch_count_larger_than_archive_returns_every_clip(tmp_path):
    project = _project(tmp_path)
    payload = _mini_archive(words=("up", "down"), per_word=2)
    paths = fetch_negatives.fetch_negatives(
        project, count=10, seed=1, downloader=_fake_downloader(payload))
    # Only 4 clips exist; the fetch returns all of them without complaining.
    assert len(paths) == 4
    assert {path.name.split("__")[0] for path in paths} == {"sc_up", "sc_down"}
