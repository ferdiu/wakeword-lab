"""Fetch real-world ambience from the DEMAND database (Zenodo record 1227121).

Stdlib only, mirroring fetch_negatives: each 16 kHz mono environment archive is
downloaded over HTTPS with an explicit timeout, verified against a pinned
SHA-256 (fail closed on mismatch), and only .wav members are extracted (with
path-traversal guards). WAVs are published into data/raw/noise/ named
``demand_<env>__<original>`` and pinned round-robin to train/validation/test in
metadata.csv so every split receives its own noise groups for noise_mixing.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .synthesize import _append_metadata

DOWNLOAD_TIMEOUT_S = 300
# DEMAND (Diverse Environments Multichannel Acoustic Noise Database), Zenodo
# record 1227121. Each archive holds the 16 kHz mono mixdowns of one
# environment (~5 minutes of continuous ambience).
URL_TEMPLATE = "https://zenodo.org/record/1227121/files/{env}_16k.zip"
DEFAULT_ENVIRONMENTS = ("DKITCHEN", "OOFFICE", "STRAFFIC")
SPLIT_ORDER = ("train", "validation", "test")
# SHA-256 of each upstream archive, measured by downloading the real zip
# (fail closed on mismatch so a corrupted or swapped download can never feed
# truncated audio into the dataset). A None entry skips verification until the
# real archive has been measured; requesting an environment missing from this
# dict fails closed before any network access.
KNOWN_SHA256: dict[str, str | None] = {
    "DKITCHEN": "9d68e46d2709847c59580770e2f8aa160607ebff7475788174aa1080332cc845",
    "OOFFICE": "7570c952f621990d0d71b8398107d9a78066254f883a5569abec9c233ffda358",
    "STRAFFIC": "2df9b444afb3b789df92d96cb6f58a50653cdbbbe7e7e74fcd9de2af6385fbea",
}


def _default_downloader(url: str, destination) -> None:
    # Per-call socket timeout via urlopen; no deprecated urlretrieve and no
    # process-global socket.setdefaulttimeout mutation.
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        with open(destination, "wb") as sink:
            shutil.copyfileobj(response, sink)


def _verify_archive_hash(env: str, archive_path: Path) -> None:
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != KNOWN_SHA256[env]:
        raise RuntimeError(
            f"{env}_16k.zip SHA-256 mismatch: got {digest}, expected {KNOWN_SHA256[env]}"
        )


def _check_member_safe(member_name: str) -> None:
    path = Path(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Refusing unsafe archive member: {member_name}")


def fetch_noise(project, environments=None, seed: int = 42, downloader=None) -> list[Path]:
    """Download DEMAND environments into data/raw/noise/ with split pins.

    Files are named ``demand_<env>__<original>`` so the environment becomes the
    source group. Metadata rows pin environment ``i`` (in sorted order) to
    ``["train", "validation", "test"][i % 3]`` so every split owns at least one
    noise group when three or more environments are fetched; users may edit the
    split column in metadata.csv afterwards.

    ``downloader`` is injectable for tests (called as ``downloader(url,
    destination_path)`` and must write a zip archive); the integrity hash is
    only enforced for the built-in downloader, since injected downloaders serve
    synthetic archives.
    """
    del seed  # Whole environments are extracted; selection is deterministic.
    environments = tuple(environments or DEFAULT_ENVIRONMENTS)
    unknown = [env for env in environments if env not in KNOWN_SHA256]
    if unknown:
        raise ValueError(
            "No pinned SHA-256 for DEMAND environment(s) "
            f"{', '.join(unknown)}; download the real archive, measure its "
            "SHA-256 and add it to KNOWN_SHA256 in fetch_noise.py"
        )
    project = Path(project)
    out_dir = project / "data" / "raw" / "noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_integrity = downloader is None
    if downloader is None:
        downloader = _default_downloader
    # Round-robin split pins over the sorted environment list.
    pins = {env: SPLIT_ORDER[index % 3] for index, env in enumerate(sorted(environments))}
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="wakeword-noise-") as temp_dir:
        for env in environments:
            url = URL_TEMPLATE.format(env=env)
            archive_path = Path(temp_dir) / f"{env}_16k.zip"
            downloader(url, archive_path)
            if verify_integrity and KNOWN_SHA256[env] is not None:
                _verify_archive_hash(env, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                members = [info for info in archive.infolist()
                           if not info.is_dir() and info.filename.lower().endswith(".wav")
                           and "__MACOSX" not in Path(info.filename).parts
                           and not Path(info.filename).name.startswith("._")]
                for info in members:
                    _check_member_safe(info.filename)
                if not members:
                    raise RuntimeError(f"Downloaded {env} archive contains no .wav entries")
                metadata_rows = []
                for info in members:
                    target = out_dir / f"demand_{env.lower()}__{Path(info.filename).name}"
                    # Atomic publish (temp + os.replace, as in fetch_negatives):
                    # an interrupted copy can never leave a partial WAV behind.
                    temporary = target.with_name(target.name + ".tmp")
                    try:
                        with archive.open(info) as source, temporary.open("wb") as sink:
                            shutil.copyfileobj(source, sink)
                        os.replace(temporary, target)
                    except Exception:
                        temporary.unlink(missing_ok=True)
                        raise
                    paths.append(target)
                    metadata_rows.append({
                        "path": target.name,
                        "speaker_id": f"demand_{env}",
                        "group_id": f"demand_{env}",
                        "split": pins[env],
                    })
                _append_metadata(project / "data/raw/metadata.csv", metadata_rows)
                total_mb = sum(p.stat().st_size for p in paths if p.name.startswith(f"demand_{env.lower()}__")) / 1e6
                print(f"Fetched {env}: {len(metadata_rows)} WAVs, {total_mb:.1f} MB "
                      f"into {out_dir} (split={pins[env]})")
    return paths
