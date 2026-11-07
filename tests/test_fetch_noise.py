import csv
import io
import zipfile

import pytest

from wakeword_lab import fetch_noise
from wakeword_lab.cli import _project_init


def _zip_bytes(entries):
    """entries: list of (archive name, payload bytes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _env_archive(env="DKITCHEN", wavs=("ch01.wav", "ch02.wav")):
    """Mirror the real DEMAND layout: WAVs nest under an <ENV>/ directory."""
    entries = [(f"{env}/{name}", b"RIFF-fake-wav") for name in wavs]
    entries.append((f"{env}/README.txt", b"not audio"))
    return _zip_bytes(entries)


def _project(tmp_path):
    project = tmp_path / "proj"
    _project_init(project, "Hey Nova")
    return project


def _fake_downloader(payloads):
    """payloads: dict of env -> zip bytes, keyed by the URL requested."""
    def _download(url, destination):
        env = url.rsplit("/", 1)[-1].removesuffix("_16k.zip")
        with open(destination, "wb") as stream:
            stream.write(payloads[env])

    return _download


def _metadata_rows(project):
    with (project / "data/raw/metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def test_fetch_noise_names_files_and_pins_round_robin(tmp_path):
    project = _project(tmp_path)
    envs = ("DKITCHEN", "OOFFICE", "STRAFFIC")
    payloads = {env: _env_archive(env) for env in envs}
    paths = fetch_noise.fetch_noise(project, environments=envs, downloader=_fake_downloader(payloads))

    assert len(paths) == 6
    names = sorted(path.name for path in paths)
    assert all(name.startswith("demand_") and "__" in name for name in names)
    # The original member name is kept after the prefix, without directories.
    assert names == [
        "demand_dkitchen__ch01.wav", "demand_dkitchen__ch02.wav",
        "demand_ooffice__ch01.wav", "demand_ooffice__ch02.wav",
        "demand_straffic__ch01.wav", "demand_straffic__ch02.wav",
    ]
    # Sorted envs are pinned round-robin across the three splits.
    rows = {row["path"]: row for row in _metadata_rows(project)}
    assert rows["demand_dkitchen__ch01.wav"]["split"] == "train"
    assert rows["demand_ooffice__ch01.wav"]["split"] == "validation"
    assert rows["demand_straffic__ch01.wav"]["split"] == "test"
    for env in ("dkitchen", "ooffice", "straffic"):
        row = rows[f"demand_{env}__ch01.wav"]
        assert row["speaker_id"] == f"demand_{env.upper()}"
        assert row["group_id"] == f"demand_{env.upper()}"


def test_fetch_noise_round_robin_wraps_beyond_three_envs(tmp_path, monkeypatch):
    project = _project(tmp_path)
    # TBUS is a real DEMAND environment without a measured hash; add it the way
    # a maintainer would after measuring the real archive.
    monkeypatch.setitem(fetch_noise.KNOWN_SHA256, "TBUS", None)
    envs = ("DKITCHEN", "OOFFICE", "STRAFFIC", "TBUS")
    payloads = {env: _env_archive(env, wavs=("ch01.wav",)) for env in envs}
    fetch_noise.fetch_noise(project, environments=envs, downloader=_fake_downloader(payloads))
    rows = {row["path"]: row for row in _metadata_rows(project)}
    assert rows["demand_dkitchen__ch01.wav"]["split"] == "train"
    assert rows["demand_ooffice__ch01.wav"]["split"] == "validation"
    assert rows["demand_straffic__ch01.wav"]["split"] == "test"
    assert rows["demand_tbus__ch01.wav"]["split"] == "train"


def test_fetch_noise_fails_closed_for_unpinned_environment(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(ValueError, match="No pinned SHA-256.*MEADOW"):
        fetch_noise.fetch_noise(project, environments=("MEADOW",),
                                downloader=_fake_downloader({}))
    assert list((project / "data/raw/noise").glob("*.wav")) == []


def test_fetch_noise_hash_mismatch_fails_closed(tmp_path):
    _project(tmp_path)
    archive = tmp_path / "DKITCHEN_16k.zip"
    archive.write_bytes(_env_archive())
    original = fetch_noise.KNOWN_SHA256["DKITCHEN"]
    fetch_noise.KNOWN_SHA256["DKITCHEN"] = "0" * 64
    try:
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            fetch_noise._verify_archive_hash("DKITCHEN", archive)
    finally:
        fetch_noise.KNOWN_SHA256["DKITCHEN"] = original


def test_fetch_noise_rejects_path_traversal_members(tmp_path):
    project = _project(tmp_path)
    payload = _zip_bytes([
        ("DKITCHEN/ch01.wav", b"RIFF-fake-wav"),
        ("../evil.wav", b"RIFF-fake-wav"),
    ])
    with pytest.raises(ValueError, match="unsafe archive member"):
        fetch_noise.fetch_noise(project, environments=("DKITCHEN",),
                                downloader=_fake_downloader({"DKITCHEN": payload}))
    assert list((project / "data/raw/noise").glob("*.wav")) == []


def test_fetch_noise_rejects_archive_without_wavs(tmp_path):
    project = _project(tmp_path)
    payload = _zip_bytes([("DKITCHEN/README.txt", b"no audio here")])
    with pytest.raises(RuntimeError, match="no .wav entries"):
        fetch_noise.fetch_noise(project, environments=("DKITCHEN",),
                                downloader=_fake_downloader({"DKITCHEN": payload}))
