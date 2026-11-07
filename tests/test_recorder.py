import re
import subprocess
from pathlib import Path

import pytest

from wakeword_lab import recorder
from wakeword_lab.cli import _project_init


def _project(tmp_path, wake_word="Hey Nova"):
    project = tmp_path / "proj"
    _project_init(project, wake_word)
    return project


def _fake_adb(files, calls, pull_rc=0):
    """Materialize a fake recordings tree when the pull lands, recording calls."""

    def _runner(args):
        calls.append(list(args))
        if args[0] == "pull":
            if pull_rc:
                return subprocess.CompletedProcess(args, pull_rc, "", "remote object does not exist")
            for rel in files:
                target = Path(args[2]) / "recordings" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"RIFF-fake-wav")
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    return _runner


def test_fetch_recordings_moves_files_and_writes_metadata(tmp_path, capsys):
    project = _project(tmp_path)
    files = [
        "positive/train/rec_20261107-120000_1500.wav",
        "positive/train/rec_20261107-120001_900.wav",
        "negative/test/rec_20261107-120002_1200.wav",
        "positive/junk/rec_20261107-120003_800.wav",  # unknown split: skipped
    ]
    calls = []
    result = recorder.fetch_recordings(project, speaker="alice", adb_runner=_fake_adb(files, calls))

    assert result["fetched"] == 3
    assert result["counts"] == {"positive": {"train": 2}, "negative": {"test": 1}}
    # One global pull index across kinds (sorted: negative first); naming scheme
    # and raw dir placement.
    positives = sorted((project / "data/raw/positive").glob("*.wav"))
    negatives = sorted((project / "data/raw/negative").glob("*.wav"))
    batch = _batch(negatives[0].name)
    assert [p.name for p in positives] == [f"alice__phone_{batch}_002.wav",
                                           f"alice__phone_{batch}_003.wav"]
    assert [p.name for p in negatives] == [f"alice__phone_{batch}_001.wav"]
    assert all(not p.name.endswith(".tmp") for p in positives + negatives)
    # Split pinned from the on-device folder; group pinned per pull batch.
    rows = _metadata_rows(project)
    assert [(r["path"], r["split"], r["group_id"]) for r in rows] == [
        (negatives[0].name, "test", f"alice_phone_{batch}"),
        (positives[0].name, "train", f"alice_phone_{batch}"),
        (positives[1].name, "train", f"alice_phone_{batch}"),
    ]
    assert all(r["speaker_id"] == "alice" and r["locale"] == "" for r in rows)
    assert "positive/train: 2" in capsys.readouterr().out


def _batch(name):
    return re.match(r"^[^_]+__phone_(\d{8}-\d{6})_\d{3}\.wav$", name).group(1)


def _metadata_rows(project):
    import csv

    with (project / "data/raw/metadata.csv").open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["path"].endswith(".wav")]


def test_fetch_recordings_delete_runs_rm_only_after_success(tmp_path):
    project = _project(tmp_path)
    files = ["positive/train/rec_a.wav", "negative/validation/rec_b.wav"]
    calls = []
    recorder.fetch_recordings(project, delete=True, adb_runner=_fake_adb(files, calls))
    assert calls[-1] == ["shell", "rm", "-rf", recorder.REMOTE_RECORDINGS]
    assert [c[0] for c in calls] == ["pull", "shell"]

    calls_fail = []
    result = recorder.fetch_recordings(project, delete=True,
                                       adb_runner=_fake_adb(files, calls_fail, pull_rc=1))
    assert result == {"fetched": 0, "counts": {}}
    assert [c[0] for c in calls_fail] == ["pull"]  # no delete after a failed pull


def test_fetch_recordings_safe_speaker_and_default(tmp_path):
    project = _project(tmp_path)
    calls = []
    recorder.fetch_recordings(project, speaker="Dr. No No",
                              adb_runner=_fake_adb(["positive/test/rec_a.wav"], calls))
    (name,) = [p.name for p in (project / "data/raw/positive").glob("*.wav")]
    assert name.startswith("Dr_No_No__phone_")


def test_fetch_recordings_errors_without_adb(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr(recorder.shutil, "which", lambda _name: None)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-sdk-here"))
    with pytest.raises(RuntimeError, match="adb not found"):
        recorder.fetch_recordings(project)


def _capture_gradle(monkeypatch, tmp_path, calls):
    def _fake_run(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(recorder.subprocess, "run", _fake_run)
    (tmp_path / "jdk17").mkdir()


def test_build_recorder_passes_wake_word_as_single_argv_item(monkeypatch, tmp_path):
    project = _project(tmp_path, wake_word='Hey "Vox"')
    calls = []
    _capture_gradle(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "jdk17"))

    apk = recorder.build_recorder(project)

    (call,) = calls
    assert call["argv"][1] == '-PwakeWord=Hey "Vox"'  # one argv item, no shell
    assert call["argv"][2:] == [":recorder:assembleDebug"]
    assert call["kwargs"]["cwd"] == recorder._repo_android_dir()
    assert call["kwargs"]["timeout"] == recorder.BUILD_TIMEOUT_S
    assert call["kwargs"]["env"]["JAVA_HOME"] == str(tmp_path / "jdk17")
    assert apk == recorder._repo_android_dir() / "recorder/build/outputs/apk/debug/recorder-debug.apk"


def test_build_recorder_install_adds_install_task(monkeypatch, tmp_path):
    project = _project(tmp_path)
    calls = []
    _capture_gradle(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "jdk17"))

    recorder.build_recorder(project, install=True)
    assert calls[0]["argv"][-2:] == [":recorder:assembleDebug", ":recorder:installDebug"]


def test_build_recorder_resolves_jdk17_from_gradle_jdks(monkeypatch, tmp_path):
    project = _project(tmp_path)
    calls = []
    _capture_gradle(monkeypatch, tmp_path, calls)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    jdk = tmp_path / "home-jdks" / ".gradle" / "jdks" / "eclipse_adoptium-17-amd64-linux.2"
    (tmp_path / "home-jdks" / ".gradle" / "jdks" / "ibm-21-jdk").mkdir(parents=True)
    jdk.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home-jdks"))

    recorder.build_recorder(project)
    assert calls[0]["kwargs"]["env"]["JAVA_HOME"] == str(jdk)


def test_build_recorder_requires_wakeword_json(tmp_path):
    with pytest.raises(FileNotFoundError, match="wakeword.json"):
        recorder.build_recorder(tmp_path / "empty")


def test_build_recorder_reads_wake_word_from_project(monkeypatch, tmp_path):
    project = _project(tmp_path, wake_word="Hey Vox")
    calls = []
    _capture_gradle(monkeypatch, tmp_path, calls)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path / "jdk17"))

    recorder.build_recorder(project)
    assert calls[0]["argv"][1] == "-PwakeWord=Hey Vox"


def test_cli_dispatches_recorder_commands(tmp_path, monkeypatch, capsys):
    project = _project(tmp_path, wake_word="Hey Vox")
    seen = {}

    def _fake_build(proj, install=False):
        seen["build"] = (Path(proj), install)
        return Path("recorder-debug.apk")

    def _fake_fetch(proj, speaker="phone", delete=False):
        seen["fetch"] = (Path(proj), speaker, delete)
        return {"fetched": 0, "counts": {}}

    monkeypatch.setattr(recorder, "build_recorder", _fake_build)
    monkeypatch.setattr(recorder, "fetch_recordings", _fake_fetch)

    from wakeword_lab.cli import main

    monkeypatch.setattr("sys.argv", ["wakeword-lab", "build-recorder", str(project), "--install"])
    main()
    assert seen["build"] == (project, True)

    monkeypatch.setattr("sys.argv", ["wakeword-lab", "fetch-recordings", str(project),
                                     "--speaker", "bob", "--delete"])
    main()
    assert seen["fetch"] == (project, "bob", True)
