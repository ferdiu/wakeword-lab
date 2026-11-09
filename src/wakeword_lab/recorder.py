"""On-device sample recorder: build the Android capture app and pull its takes.

Stdlib only. ``build_recorder`` compiles the standalone ``android/recorder``
app with the project wake word baked in; ``fetch_recordings`` pulls the
recorded WAV tree over adb into ``data/raw/<kind>/`` and tracks every take in
``metadata.csv`` with the split pinned from the on-device folder.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .synthesize import _append_metadata

RECORDER_PACKAGE = "dev.wakeword.recorder"
REMOTE_RECORDINGS = f"/sdcard/Android/data/{RECORDER_PACKAGE}/files/recordings"
DEVICE_PULL_TIMEOUT_S = 300
DEVICE_DELETE_TIMEOUT_S = 60
BUILD_TIMEOUT_S = 900
VALID_KINDS = ("positive", "negative")
VALID_SPLITS = ("train", "validation", "test")


def _repo_android_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "android"


def _safe_speaker(speaker: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", speaker).strip("_") or "unknown"


def _resolve_java_home(environ=None, home=None) -> str:
    """JAVA_HOME from the environment wins; otherwise pick a JDK 17 from
    ~/.gradle/jdks (Gradle-provisioned toolchains), else fail loudly."""
    environ = os.environ if environ is None else environ
    if environ.get("JAVA_HOME"):
        return environ["JAVA_HOME"]
    jdks_root = Path(home if home is not None else Path.home()) / ".gradle" / "jdks"
    candidates = sorted(path for path in jdks_root.iterdir() if path.is_dir()) if jdks_root.is_dir() else []
    for candidate in candidates:
        if "17" in candidate.name:
            return str(candidate)
    raise RuntimeError(
        "JAVA_HOME is not set and no JDK 17 was found under "
        f"{jdks_root}. Export JAVA_HOME or install a Gradle-provisioned JDK 17."
    )


def build_recorder(project, install: bool = False) -> Path:
    """Compile the recorder APK (and optionally install it) for ``project``.

    The wake word is read from ``<project>/wakeword.json`` and passed to
    Gradle as a single ``-PwakeWord=...`` argv item (no shell, so arbitrary
    phrases are safe). Returns the debug APK path.
    """
    project = Path(project)
    config_path = project / "wakeword.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found; run `wakeword-lab init` first")
    wake_word = json.loads(config_path.read_text(encoding="utf-8"))["wake_word"]
    android_dir = _repo_android_dir()
    gradlew = android_dir / ("gradlew.bat" if os.name == "nt" else "gradlew")
    if not gradlew.exists():
        raise FileNotFoundError(f"{gradlew} not found; is the android/ checkout complete?")
    argv = [str(gradlew), f"-PwakeWord={wake_word}", ":recorder:assembleDebug"]
    if install:
        argv.append(":recorder:installDebug")
    env = dict(os.environ, JAVA_HOME=_resolve_java_home())
    subprocess.run(argv, cwd=android_dir, timeout=BUILD_TIMEOUT_S, env=env, check=True)
    apk = android_dir / "recorder" / "build" / "outputs" / "apk" / "debug" / "recorder-debug.apk"
    print(f"Recorder APK: {apk}")
    return apk


def _find_adb() -> str:
    found = shutil.which("adb")
    if found:
        return found
    candidates = []
    android_home = os.environ.get("ANDROID_HOME")
    if android_home:
        candidates.append(Path(android_home) / "platform-tools" / "adb")
    candidates.append(Path.home() / ".android-sdk" / "sdk" / "platform-tools" / "adb")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    searched = " or ".join(str(c) for c in candidates)
    raise RuntimeError(
        "adb not found on PATH" + (f" or at {searched}" if searched else "") +
        "; install the Android SDK platform-tools to fetch recordings"
    )


def fetch_recordings(project, speaker: str = "phone", delete: bool = False, adb_runner=None) -> dict:
    """Pull recordings/<kind>/<split>/*.wav from the device into data/raw/<kind>/.

    ``adb_runner`` is injectable for tests: called as ``runner(list_of_args)``
    and must return a ``subprocess.CompletedProcess``. Each take is published
    as ``<safe_speaker>__phone_<batch>_<NNN>.wav`` (batch = pull timestamp,
    NNN = global index within this pull) and pinned in metadata.csv to the
    split of its on-device folder, grouped per pull and split as
    ``<speaker>_phone_<batch>_<split>`` (one group must never straddle
    partitions). Unknown kind/split folders are skipped with a
    warning. With ``delete`` the on-device tree is removed only after every
    file moved successfully. Returns ``{"fetched": n, "counts": {kind: {split: n}}}``.
    """
    project = Path(project)
    if adb_runner is not None:
        pull_runner = delete_runner = adb_runner
    else:
        adb = _find_adb()

        def pull_runner(args):
            return subprocess.run([adb, *args], capture_output=True, text=True,
                                  timeout=DEVICE_PULL_TIMEOUT_S)

        def delete_runner(args):
            return subprocess.run([adb, *args], capture_output=True, text=True,
                                  timeout=DEVICE_DELETE_TIMEOUT_S)

    with tempfile.TemporaryDirectory(prefix="wakeword-recordings-") as temp_dir:
        pull = pull_runner(["pull", REMOTE_RECORDINGS, temp_dir])
        if pull.returncode != 0:
            print("No recordings pulled (is the device connected and the recorder app used?) "
                  f"— adb said: {(pull.stderr or pull.stdout or '').strip()}")
            return {"fetched": 0, "counts": {}}
        batch = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_speaker = _safe_speaker(speaker)
        metadata_path = project / "data" / "raw" / "metadata.csv"
        tree = Path(temp_dir) / "recordings"
        counts: dict[str, dict[str, int]] = {}
        moved = 0
        for kind_dir in sorted(tree.iterdir()) if tree.is_dir() else []:
            kind = kind_dir.name
            if kind not in VALID_KINDS:
                print(f"warning: skipping unknown folder {kind_dir.relative_to(tree)}")
                continue
            out_dir = project / "data" / "raw" / kind
            out_dir.mkdir(parents=True, exist_ok=True)
            for split_dir in sorted(kind_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                if split_dir.name not in VALID_SPLITS:
                    print(f"warning: skipping unknown folder {split_dir.relative_to(tree)}")
                    continue
                split = split_dir.name
                # Split integrity: one group must never straddle partitions, so
                # the pull-batch group is per split, not per pull.
                group_id = f"{speaker}_phone_{batch}_{split}"
                for wav in sorted(split_dir.glob("*.wav")):
                    moved += 1
                    target = out_dir / f"{safe_speaker}__phone_{batch}_{moved:03d}.wav"
                    # Atomic publish (temp + os.replace, as in fetch_negatives):
                    # an interrupted move can never leave a partial WAV behind.
                    temporary = target.with_name(target.name + ".tmp")
                    try:
                        shutil.copyfile(wav, temporary)
                        os.replace(temporary, target)
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        raise
                    wav.unlink()
                    try:
                        _append_metadata(metadata_path, [{
                            "path": target.name, "speaker_id": speaker, "group_id": group_id,
                            "split": split, "locale": "", "phrase_start_s": "", "phrase_end_s": "",
                        }])
                    except Exception:
                        target.unlink(missing_ok=True)
                        raise
                    bucket = counts.setdefault(kind, {})
                    bucket[split] = bucket.get(split, 0) + 1
    if delete and moved:
        delete_runner(["shell", "rm", "-rf", REMOTE_RECORDINGS])
    print(f"Fetched {moved} recordings into {project / 'data' / 'raw'}:")
    for kind in sorted(counts):
        for split in sorted(counts[kind]):
            print(f"  {kind}/{split}: {counts[kind][split]}")
    return {"fetched": moved, "counts": counts}
