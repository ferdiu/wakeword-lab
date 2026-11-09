import io
import tarfile
import wave
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


def _tar_gz_bytes(entries):
    """entries: list of (archive name, payload bytes)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
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


def _full_archive(words=("up", "down"), per_word=4):
    """Mirror the real speech_commands_v0.02 layout: <word>/<clip>.wav at the
    archive root (the real tar prefixes members with ./), a
    _background_noise_ folder, and a non-WAV member."""
    entries = []
    for word in words:
        for index in range(per_word):
            # The real archive prefixes members with "./"; mix both forms.
            prefix = "./" if index % 2 else ""
            entries.append((f"{prefix}{word}/hash_nohash_{index}.wav", b"RIFF-fake-wav"))
    for index in range(3):
        entries.append((f"./_background_noise_/noise_{index}.wav", b"RIFF-fake-wav"))
    entries.append(("./README.md", b"not audio"))
    return _tar_gz_bytes(entries)


def test_fetch_full_selects_balanced_clips_and_skips_background_noise(tmp_path, capsys):
    project = _project(tmp_path)
    payload = _full_archive(words=("up", "down", "left"), per_word=4)
    paths = fetch_negatives.fetch_negatives(
        project, count=6, seed=1, source="speech-commands-full",
        downloader=_fake_downloader(payload))

    assert len(paths) == 6
    names = sorted(path.name for path in paths)
    assert all(name.startswith("sc_") and "__" in name for name in names)
    # Uniform across word folders: 3 words x 2 clips each.
    per_word = {}
    for name in names:
        per_word[name[len("sc_"):].split("__")[0]] = per_word.get(name[len("sc_"):].split("__")[0], 0) + 1
    assert per_word == {"up": 2, "down": 2, "left": 2}
    # _background_noise_ must never become a group; the README is not audio.
    assert not any("background_noise" in name or "README" in name for name in names)
    summary = capsys.readouterr().out
    assert "up: 2" in summary and "down: 2" in summary and "left: 2" in summary


def test_fetch_full_rejects_path_traversal_members(tmp_path):
    project = _project(tmp_path)
    payload = _tar_gz_bytes([
        ("./up/hash_nohash_0.wav", b"RIFF-fake-wav"),
        ("../evil.wav", b"RIFF-fake-wav"),
    ])
    with pytest.raises(ValueError, match="unsafe archive member"):
        fetch_negatives.fetch_negatives(
            project, count=2, seed=1, source="speech-commands-full",
            downloader=_fake_downloader(payload))
    assert list((project / "data/raw/negative").glob("*.wav")) == []


def test_fetch_full_count_larger_than_archive_returns_every_clip(tmp_path):
    project = _project(tmp_path)
    payload = _full_archive(words=("up", "down"), per_word=2)
    paths = fetch_negatives.fetch_negatives(
        project, count=50, seed=1, source="speech-commands-full",
        downloader=_fake_downloader(payload))
    # 4 clips exist (background noise excluded); all returned, no exception.
    assert len(paths) == 4
    assert {path.name.split("__")[0] for path in paths} == {"sc_up", "sc_down"}


def _librispeech_archive(speakers=("111", "222"), per_chapter=3):
    """Mirror the real test-clean layout:
    LibriSpeech/test-clean/<speaker>/<chapter>/<clip>.flac plus shallow
    non-FLAC files."""
    entries = [("LibriSpeech/README.TXT", b"not audio")]
    for speaker in speakers:
        for index in range(per_chapter):
            entries.append(
                (f"LibriSpeech/test-clean/{speaker}/11/{speaker}-{index}.flac",
                 b"fLaC-fake"))
    return _tar_gz_bytes(entries)


def _tiny_wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


def _fake_ffmpeg(monkeypatch):
    """Pretend ffmpeg converts FLAC to WAV by writing a tiny valid WAV."""
    calls = []

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        with open(argv[-1], "wb") as sink:
            sink.write(_tiny_wav_bytes())

    monkeypatch.setattr(fetch_negatives.subprocess, "run", _run)
    return calls


def test_fetch_librispeech_converts_flac_with_speaker_prefix_names(tmp_path, monkeypatch):
    project = _project(tmp_path)
    calls = _fake_ffmpeg(monkeypatch)
    payload = _librispeech_archive(speakers=("111", "222"), per_chapter=3)
    paths = fetch_negatives.fetch_negatives(
        project, count=4, seed=1, source="librispeech",
        downloader=_fake_downloader(payload))

    assert len(paths) == 4
    names = sorted(path.name for path in paths)
    # FLAC originals are renamed to .wav; speakers are the balance groups.
    assert all(name.startswith("ls_") and name.endswith(".wav") for name in names)
    assert not any(name.endswith(".flac") for name in names)
    per_speaker = {}
    for name in names:
        speaker = name[len("ls_"):].split("__")[0]
        per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
    assert per_speaker == {"111": 2, "222": 2}  # uniform across speaker folders
    # ffmpeg was invoked once per selected clip with the conversion argv shape.
    assert len(calls) == 4
    argv, kwargs = calls[0]
    assert argv[0] == "ffmpeg" and "-nostdin" in argv and "-loglevel" in argv
    assert argv[argv.index("-ar") + 1] == "16000"
    assert kwargs["timeout"] == fetch_negatives.FFMPEG_TIMEOUT_S
    # Each converted output is a real WAV on disk.
    for path in paths:
        with wave.open(str(path)) as stream:
            assert stream.getframerate() == 16000 and stream.getnchannels() == 1


def test_fetch_librispeech_requires_ffmpeg(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr(fetch_negatives.shutil, "which", lambda name: None)
    payload = _librispeech_archive()
    with pytest.raises(RuntimeError, match="ffmpeg executable not found"):
        fetch_negatives.fetch_negatives(
            project, count=2, seed=1, source="librispeech",
            downloader=_fake_downloader(payload))
    assert list((project / "data/raw/negative").glob("*.wav")) == []


def test_rerun_with_different_source_keeps_other_source_files(tmp_path, monkeypatch):
    project = _project(tmp_path)
    _fake_ffmpeg(monkeypatch)
    fetch_negatives.fetch_negatives(
        project, count=2, seed=1, downloader=_fake_downloader(_mini_archive(per_word=2)))
    fetch_negatives.fetch_negatives(
        project, count=2, seed=1, source="librispeech",
        downloader=_fake_downloader(_librispeech_archive(per_chapter=2)))
    names = sorted(p.name for p in (project / "data/raw/negative").glob("*.wav"))
    assert sum(name.startswith("sc_") for name in names) == 2
    assert sum(name.startswith("ls_") for name in names) == 2
