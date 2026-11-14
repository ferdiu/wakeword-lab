"""Fetch hard-negative speech clips from public archives.

Stdlib only. Downloads an archive over HTTPS with an explicit timeout,
verifies its pinned SHA-256 (fail closed on mismatch), extracts only audio
members (with path-traversal guards) and selects a balanced set uniformly
across the archive's top-level group folders — command words for the
Speech Commands datasets, speaker IDs for LibriSpeech — so the
filename-prefix grouping keeps every group inside one split partition.

Supported sources:

- ``speech-commands-mini``: TensorFlow mini_speech_commands zip (~174 MB),
  16 kHz mono WAVs copied as ``sc_<word>__<original>.wav``.
- ``speech-commands-full``: TensorFlow speech_commands_v0.02 tar.gz
  (~2.3 GB), 16 kHz mono WAVs copied as ``sc_<word>__<original>.wav``;
  the ``_background_noise_`` folder is excluded (DEMAND covers ambience).
- ``librispeech``: OpenSLR LibriSpeech test-clean tar.gz (~350 MB), FLAC
  clips converted with ffmpeg to 16 kHz mono WAV named
  ``ls_<speaker>__<original>.wav`` so each speaker stays in one partition.
- ``librispeech-train``: OpenSLR LibriSpeech train-clean-100 tar.gz
  (~6.3 GB), same FLAC-to-WAV handling named
  ``lt_<speaker>__<original>.wav``.
- ``musan-speech``: OpenSLR MUSAN tar.gz (~11 GB, one archive shared with
  ``musan-music``), 16 kHz mono WAVs copied as
  ``ms_<collection>__<original>.wav``; only the ``musan/speech`` subtree is
  selected — ``musan/music`` belongs to ``musan-music`` and ``musan/noise``
  to ``fetch-noise``.
- ``musan-music``: the same MUSAN archive, long music clips from the
  ``musan/music`` subtree copied as ``mm_<collection>__<original>.wav``.
- ``esc-50``: GitHub ESC-50 zip (~600 MB), 5 s 44.1 kHz stereo WAVs
  converted with ffmpeg to 16 kHz mono named ``es_<category>__<original>.wav``;
  the balance group is the sound category read from ``meta/esc50.csv``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DOWNLOAD_TIMEOUT_S = 300
FFMPEG_TIMEOUT_S = 60
DEFAULT_SOURCE = "speech-commands-mini"


@dataclass(frozen=True)
class SourceSpec:
    """Static description of one negative-speech archive."""
    url: str
    archive_name: str          # local temp filename for the downloaded archive
    sha256: str | None         # pinned upstream digest; None = unmeasured TODO
    default_count: int
    prefix: str                # target filename prefix ("sc" or "ls")
    audio_suffix: str          # member suffix that counts as audio
    convert_to_wav: bool       # True: ffmpeg-convert extracted audio to WAV
    group_of: Callable[[str], str | None]
    # group_of maps a member name to its balance/group folder, or None to
    # skip the member (out of scope, e.g. _background_noise_ or a non-clip
    # path such as LibriSpeech's top-level README/license files).
    group_map_reader: Callable[[zipfile.ZipFile], dict[str, str]] | None = None
    # group_map_reader (zip sources only) reads archive metadata into a
    # member-name -> group map that takes precedence over group_of; esc-50
    # uses it to group audio members by their esc50.csv sound category.


# SHA-256 values were measured by downloading the real upstream archives
# (fail closed on mismatch so a corrupted or swapped download can never feed
# truncated audio into the dataset). A None entry skips verification until
# the real archive has been measured; see fetch_noise.KNOWN_SHA256 for the
# same pattern.
def _mini_group(member_name: str) -> str | None:
    # The real archive nests clips under a top-level mini_speech_commands/
    # directory, so the command word is the member's parent directory.
    parts = Path(member_name).parts
    return parts[-2] if len(parts) > 1 else "_root"


def _full_group(member_name: str) -> str | None:
    # Members are <word>/<hash>_nohash_<n>.wav at the archive root; the
    # _background_noise_ folder is DEMAND's job, so its members are skipped.
    parts = Path(member_name).parts
    if len(parts) < 2:
        return None
    word = parts[0]
    return None if word == "_background_noise_" else word


def _librispeech_group(member_name: str) -> str | None:
    # Members are LibriSpeech/<subset>/<speaker>/<chapter>/<clip>.flac;
    # the speaker ID is the balance group. Anything shallower (README,
    # LICENSE, directory entries) is not a clip.
    parts = Path(member_name).parts
    return parts[2] if len(parts) >= 4 else None


def _musan_speech_group(member_name: str) -> str | None:
    # Members are musan/speech/<collection>/<clip>.wav; the collection
    # (language/corpus folder, e.g. us-gov, librivox) is the balance group.
    # The musan/music subtree belongs to musan-music and musan/noise to
    # fetch-noise, so only the speech subtree is selected here.
    parts = Path(member_name).parts
    if len(parts) >= 4 and parts[0] == "musan" and parts[1] == "speech":
        return parts[2]
    return None


def _musan_music_group(member_name: str) -> str | None:
    # Same archive as musan-speech: musan/music/<collection>/<clip>.wav;
    # only the music subtree is in scope for this source.
    parts = Path(member_name).parts
    if len(parts) >= 4 and parts[0] == "musan" and parts[1] == "music":
        return parts[2]
    return None


def _esc50_group(member_name: str) -> str | None:
    # esc-50 grouping comes from meta/esc50.csv via _esc50_group_map, never
    # from member paths; every path is skipped here.
    return None


def _esc50_group_map(archive: zipfile.ZipFile) -> dict[str, str]:
    """Map ESC-50-master/audio/<file> members to their esc50.csv category.

    Keys are full member names, so members outside the audio/ tree can never
    be grouped — anything not exactly an audio/ member is skipped.
    """
    with archive.open("ESC-50-master/meta/esc50.csv") as source:
        rows = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"))
        return {f"ESC-50-master/audio/{row['filename']}": row["category"]
                for row in rows}


SOURCES: dict[str, SourceSpec] = {
    "speech-commands-mini": SourceSpec(
        url="https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        archive_name="mini_speech_commands.zip",
        sha256="49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0",
        default_count=200,
        prefix="sc",
        audio_suffix=".wav",
        convert_to_wav=False,
        group_of=_mini_group,
    ),
    "speech-commands-full": SourceSpec(
        url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        archive_name="speech_commands_v0.02.tar.gz",
        # Measured 2026-11-09 from the real download (2428923189 bytes).
        sha256="af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",
        default_count=1000,
        prefix="sc",
        audio_suffix=".wav",
        convert_to_wav=False,
        group_of=_full_group,
    ),
    "librispeech": SourceSpec(
        url="https://www.openslr.org/resources/12/test-clean.tar.gz",
        archive_name="test-clean.tar.gz",
        # Measured 2026-11-09 from the real download (346663984 bytes).
        sha256="39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
        default_count=200,
        prefix="ls",
        audio_suffix=".flac",
        convert_to_wav=True,
        group_of=_librispeech_group,
    ),
    "librispeech-train": SourceSpec(
        url="https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        archive_name="train-clean-100.tar.gz",
        # Measured 2026-11-13 from the real download (6387309499 bytes).
        sha256="d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
        default_count=500,
        prefix="lt",
        audio_suffix=".flac",
        convert_to_wav=True,
        group_of=_librispeech_group,
    ),
    "musan-speech": SourceSpec(
        url="https://openslr.trmal.net/resources/17/musan.tar.gz",
        archive_name="musan.tar.gz",
        # Measured 2026-11-13 from the real download (11086114085 bytes);
        # musan-music pins the same archive (same URL, same digest).
        sha256="86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1",
        default_count=400,
        prefix="ms",
        audio_suffix=".wav",
        convert_to_wav=False,
        group_of=_musan_speech_group,
    ),
    "musan-music": SourceSpec(
        url="https://openslr.trmal.net/resources/17/musan.tar.gz",
        archive_name="musan.tar.gz",
        # Same archive as musan-speech (11086114085 bytes).
        sha256="86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1",
        default_count=300,
        prefix="mm",
        audio_suffix=".wav",
        convert_to_wav=False,
        group_of=_musan_music_group,
    ),
    "esc-50": SourceSpec(
        url="https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        archive_name="esc-50.zip",
        # Measured 2026-11-13 from the real download (645695005 bytes).
        sha256="805afc618aff80eff0641d51311647871fead5807da68bb18e43a27db47a79ce",
        default_count=400,
        prefix="es",
        audio_suffix=".wav",
        convert_to_wav=True,  # 44.1 kHz stereo -> 16 kHz mono WAV
        group_of=_esc50_group,
        group_map_reader=_esc50_group_map,
    ),
}


def _default_downloader(url: str, destination) -> None:
    # Per-call socket timeout via urlopen; no deprecated urlretrieve and no
    # process-global socket.setdefaulttimeout mutation.
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        with open(destination, "wb") as sink:
            shutil.copyfileobj(response, sink)


def _verify_archive_hash(spec: SourceSpec, archive_path: Path) -> None:
    # Stream the digest: multi-GB archives (MUSAN is ~11 GB) must never be
    # loaded into RAM whole.
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise RuntimeError(
            f"{spec.archive_name} SHA-256 mismatch: got {digest.hexdigest()}, "
            f"expected {spec.sha256}"
        )


def _check_member_safe(member_name: str) -> None:
    path = Path(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Refusing unsafe archive member: {member_name}")


def _select_balanced(by_group: dict[str, list], count: int, rng: random.Random) -> list:
    """Walk the group folders round-robin, popping a random remaining member
    from each so no group is over-represented in the selection."""
    groups = sorted(by_group)
    selected: list = []
    while len(selected) < count and any(by_group.values()):
        for group in groups:
            bucket = by_group[group]
            if bucket and len(selected) < count:
                selected.append((group, bucket.pop(rng.randrange(len(bucket)))))
    return selected


def _publish(source, target: Path) -> None:
    # Atomic publish (temp + os.replace, as in synthesize/record): an
    # interrupted copy can never leave a partial WAV behind.
    temporary = target.with_name(target.name + ".tmp")
    try:
        with source, temporary.open("wb") as sink:
            shutil.copyfileobj(source, sink)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _convert_to_wav(input_path: Path, wav_path: Path) -> None:
    """Convert one extracted audio file (FLAC or WAV — ffmpeg probes the
    input format) to 16 kHz mono pcm_s16le WAV."""
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(wav_path),
    ], check=True, timeout=FFMPEG_TIMEOUT_S)


def fetch_negatives(project, count: int | None = None, seed: int = 42,
                    source: str = DEFAULT_SOURCE, url: str | None = None,
                    downloader=None) -> list[Path]:
    """Copy ``count`` clips from ``source`` into data/raw/negative/.

    Files are named ``<prefix>_<group>__<original>`` so the default
    filename-prefix grouping keeps every group inside one split partition.
    No metadata rows are added: the README documents filename-prefix grouping
    as the default. ``count=None`` selects the source's default count.
    ``downloader`` is injectable for tests (called as ``downloader(url,
    destination_path)`` and must write an archive); the integrity hash is
    only enforced for the built-in downloader, since injected downloaders
    serve synthetic archives.
    """
    spec = SOURCES.get(source)
    if spec is None:
        raise ValueError(
            f"Unknown source {source!r}; choose from {', '.join(sorted(SOURCES))}"
        )
    if count is None:
        count = spec.default_count
    if count < 1:
        raise ValueError("count must be at least one")
    project = Path(project)
    out_dir = project / "data" / "raw" / "negative"
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_integrity = downloader is None
    if downloader is None:
        downloader = _default_downloader
    if spec.convert_to_wav and not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg executable not found; install ffmpeg to convert archive "
            "audio to 16 kHz WAV"
        )
    rng = random.Random(seed)
    paths: list[Path] = []
    per_group: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="wakeword-fetch-") as temp_dir:
        temp_dir = Path(temp_dir)
        archive_path = temp_dir / spec.archive_name
        downloader(url or spec.url, archive_path)
        if verify_integrity and spec.sha256 is not None:
            _verify_archive_hash(spec, archive_path)
        if zipfile.is_zipfile(archive_path):
            selected = _select_from_zip(archive_path, spec, count, rng)
            paths, per_group = _publish_zip_selection(
                archive_path, selected, spec, out_dir, temp_dir / "extract")
        else:
            selected = _select_from_tar(archive_path, spec, count, rng)
            paths, per_group = _publish_tar_selection(
                archive_path, selected, spec, out_dir, temp_dir / "extract")
    print(f"Fetched {len(paths)} clips into {out_dir}:")
    for group in sorted(per_group):
        if per_group[group]:
            print(f"  {group}: {per_group[group]}")
    return paths


def _bucket_members(member_names: list[str], spec: SourceSpec,
                    group_map: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Group in-scope, safe audio member names by their balance group."""
    by_group: dict[str, list[str]] = {}
    for name in member_names:
        _check_member_safe(name)
        # Archive-metadata grouping (esc-50's esc50.csv categories) takes
        # precedence over path-based grouping; unknown members are skipped.
        group = group_map.get(name) if group_map is not None else spec.group_of(name)
        if group is not None:
            by_group.setdefault(group, []).append(name)
    if not by_group:
        raise RuntimeError(
            f"Downloaded archive contains no {spec.audio_suffix} entries")
    return by_group


def _target_name(spec: SourceSpec, group: str, member_name: str) -> str:
    original = Path(member_name).name
    if spec.convert_to_wav:
        # FLAC originals change extension when converted to WAV.
        original = Path(original).stem + ".wav"
    return f"{spec.prefix}_{group}__{original}"


def _select_from_zip(archive_path: Path, spec: SourceSpec, count: int,
                     rng: random.Random) -> list[tuple[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        group_map = spec.group_map_reader(archive) if spec.group_map_reader else None
        # Skip __MACOSX/ AppleDouble junk ("._*.wav") present in the real
        # mini archive: half of its .wav members are metadata, not audio.
        names = [info.filename for info in archive.infolist()
                 if not info.is_dir() and info.filename.lower().endswith(spec.audio_suffix)
                 and "__MACOSX" not in Path(info.filename).parts
                 and not Path(info.filename).name.startswith("._")]
        by_group = _bucket_members(names, spec, group_map)
        buckets = {group: list(names) for group, names in by_group.items()}
        return _select_balanced(buckets, count, rng)


def _select_from_tar(archive_path: Path, spec: SourceSpec, count: int,
                     rng: random.Random) -> list[tuple[str, str]]:
    with tarfile.open(archive_path, mode="r:*") as archive:
        names = [info.name for info in archive.getmembers()
                 if info.isfile() and info.name.lower().endswith(spec.audio_suffix)]
        by_group = _bucket_members(names, spec)
        buckets = {group: list(names) for group, names in by_group.items()}
        return _select_balanced(buckets, count, rng)


def _publish_zip_selection(archive_path: Path, selected: list[tuple[str, str]],
                           spec: SourceSpec, out_dir: Path,
                           extract_dir: Path) -> tuple[list[Path], dict[str, int]]:
    paths: list[Path] = []
    per_group: dict[str, int] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for group, member_name in selected:
            target = out_dir / _target_name(spec, group, member_name)
            if spec.convert_to_wav:
                # ffmpeg needs a real input file; stage the member in temp
                # first (same pattern as the tar publish path).
                staged = extract_dir / f"{target.stem}{Path(member_name).suffix}"
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member_name) as source, staged.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                # ffmpeg infers the output muxer from the extension, so the
                # conversion temp file must end in .wav (as in synthesize).
                temporary = target.with_suffix(".tmp.wav")
                try:
                    _convert_to_wav(staged, temporary)
                    os.replace(temporary, target)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                finally:
                    staged.unlink(missing_ok=True)
            else:
                _publish(archive.open(member_name), target)
            per_group[group] = per_group.get(group, 0) + 1
            paths.append(target)
    return paths, per_group


def _publish_tar_selection(archive_path: Path, selected: list[tuple[str, str]],
                           spec: SourceSpec, out_dir: Path,
                           extract_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """Publish selected members in ONE streaming pass.

    tar.gz has no seekable index: random-access mode (r:*) re-decompresses the
    whole archive from the start for every extract() call, which turns 1000
    selections in a 2.4 GB archive into an hour of CPU. Streaming mode (r|*)
    decompresses once, in member order.
    """
    paths: list[Path] = []
    per_group: dict[str, int] = {}
    wanted = {member_name: group for group, member_name in selected}
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r|*") as archive:
        for info in archive:
            if not info.isfile() or info.name not in wanted:
                continue
            _check_member_safe(info.name)
            group = wanted.pop(info.name)
            target = out_dir / _target_name(spec, group, info.name)
            if spec.convert_to_wav:
                # ffmpeg needs a real input file; stage the clip in temp first.
                staged = extract_dir / f"{target.stem}{Path(info.name).suffix}"
                with archive.extractfile(info) as source, staged.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                # ffmpeg infers the output muxer from the extension, so the
                # conversion temp file must end in .wav (as in synthesize).
                temporary = target.with_suffix(".tmp.wav")
                try:
                    _convert_to_wav(staged, temporary)
                    os.replace(temporary, target)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                finally:
                    staged.unlink(missing_ok=True)
            else:
                _publish(archive.extractfile(info), target)
            per_group[group] = per_group.get(group, 0) + 1
            paths.append(target)
    if wanted:
        raise RuntimeError(
            f"{len(wanted)} selected members were not found in the archive")
    return paths, per_group
