"""Fetch command-word speech clips from the mini_speech_commands archive as hard negatives.

Stdlib only. Downloads the ~78 MB TensorFlow archive with an explicit socket
timeout, extracts only .wav members (with path-traversal guards) and selects a
balanced set uniformly across word folders.
"""
from __future__ import annotations

import random
import shutil
import socket
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
DOWNLOAD_TIMEOUT_S = 300


def _default_downloader(url: str, destination) -> None:
    # urlretrieve has no per-call timeout; cap the whole download so a stalled
    # connection cannot hang the pipeline indefinitely.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(DOWNLOAD_TIMEOUT_S)
    try:
        urllib.request.urlretrieve(url, destination)
    finally:
        socket.setdefaulttimeout(previous)


def _word_of(member_name: str) -> str:
    parts = Path(member_name).parts
    return parts[0] if len(parts) > 1 else "_root"


def _check_member_safe(member_name: str) -> None:
    path = Path(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Refusing unsafe archive member: {member_name}")


def fetch_negatives(project, count: int = 200, seed: int = 42, url: str = DEFAULT_URL,
                    downloader=None) -> list[Path]:
    """Copy ``count`` clips from the archive into data/raw/negative/.

    Files are named ``sc_<word>__<original>`` so the default filename-prefix
    grouping keeps every command word inside one split partition. No metadata
    rows are added: the README documents filename-prefix grouping as the
    default. ``downloader`` is injectable for tests (called as
    ``downloader(url, destination_path)`` and must write a zip archive).
    """
    if count < 1:
        raise ValueError("count must be at least one")
    project = Path(project)
    out_dir = project / "data" / "raw" / "negative"
    out_dir.mkdir(parents=True, exist_ok=True)
    if downloader is None:
        downloader = _default_downloader
    rng = random.Random(seed)
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="wakeword-fetch-") as temp_dir:
        archive_path = Path(temp_dir) / "mini_speech_commands.zip"
        downloader(url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = [info for info in archive.infolist()
                       if not info.is_dir() and info.filename.lower().endswith(".wav")]
            for info in members:
                _check_member_safe(info.filename)
            by_word: dict[str, list[zipfile.ZipInfo]] = {}
            for info in members:
                by_word.setdefault(_word_of(info.filename), []).append(info)
            if not by_word:
                raise RuntimeError("Downloaded archive contains no .wav entries")
            words = sorted(by_word)
            # Balanced selection: walk the word folders round-robin, popping a
            # random remaining clip from each so no word is over-represented.
            selected: list[zipfile.ZipInfo] = []
            while len(selected) < count and any(by_word.values()):
                for word in words:
                    bucket = by_word[word]
                    if bucket and len(selected) < count:
                        selected.append(bucket.pop(rng.randrange(len(bucket))))
            per_word: dict[str, int] = {word: 0 for word in words}
            for info in selected:
                word = _word_of(info.filename)
                target = out_dir / f"sc_{word}__{Path(info.filename).name}"
                with archive.open(info) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                per_word[word] += 1
                paths.append(target)
    print(f"Fetched {len(paths)} clips into {out_dir}:")
    for word in sorted(per_word):
        if per_word[word]:
            print(f"  {word}: {per_word[word]}")
    return paths
