"""Fetch command-word speech clips from the mini_speech_commands archive as hard negatives.

Stdlib only. Downloads the ~174 MB TensorFlow archive over HTTPS with an
explicit timeout, verifies its SHA-256, extracts only .wav members (with
path-traversal guards) and selects a balanced set uniformly across word folders.
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
DOWNLOAD_TIMEOUT_S = 300
# SHA-256 of the upstream archive (measured 2026-11-06, 182082353 bytes).
# Verification fails closed on mismatch so a corrupted or swapped download can
# never feed truncated audio into the dataset.
KNOWN_SHA256 = "49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0"


def _default_downloader(url: str, destination) -> None:
    # Per-call socket timeout via urlopen; no deprecated urlretrieve and no
    # process-global socket.setdefaulttimeout mutation.
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        with open(destination, "wb") as sink:
            shutil.copyfileobj(response, sink)


def _verify_archive_hash(archive_path: Path) -> None:
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != KNOWN_SHA256:
        raise RuntimeError(
            "mini_speech_commands.zip SHA-256 mismatch: "
            f"got {digest}, expected {KNOWN_SHA256}"
        )


def _word_of(member_name: str) -> str:
    # The real archive nests clips under a top-level mini_speech_commands/
    # directory, so the command word is the member's parent directory.
    parts = Path(member_name).parts
    return parts[-2] if len(parts) > 1 else "_root"


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
    default.     ``downloader`` is injectable for tests (called as
    ``downloader(url, destination_path)`` and must write a zip archive); the
    integrity hash is only enforced for the built-in downloader, since
    injected downloaders serve synthetic archives.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    project = Path(project)
    out_dir = project / "data" / "raw" / "negative"
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_integrity = downloader is None
    if downloader is None:
        downloader = _default_downloader
    rng = random.Random(seed)
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="wakeword-fetch-") as temp_dir:
        archive_path = Path(temp_dir) / "mini_speech_commands.zip"
        downloader(url, archive_path)
        if verify_integrity and KNOWN_SHA256 is not None:
            _verify_archive_hash(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            # Skip __MACOSX/ AppleDouble junk ("._*.wav") present in the real
            # archive: half of its .wav members are metadata, not audio.
            members = [info for info in archive.infolist()
                       if not info.is_dir() and info.filename.lower().endswith(".wav")
                       and "__MACOSX" not in Path(info.filename).parts
                       and not Path(info.filename).name.startswith("._")]
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
                # Atomic publish (temp + os.replace, as in synthesize/record):
                # an interrupted copy can never leave a partial WAV behind.
                temporary = target.with_name(target.name + ".tmp")
                try:
                    with archive.open(info) as source, temporary.open("wb") as sink:
                        shutil.copyfileobj(source, sink)
                    os.replace(temporary, target)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                per_word[word] += 1
                paths.append(target)
    print(f"Fetched {len(paths)} clips into {out_dir}:")
    for word in sorted(per_word):
        if per_word[word]:
            print(f"  {word}: {per_word[word]}")
    return paths
