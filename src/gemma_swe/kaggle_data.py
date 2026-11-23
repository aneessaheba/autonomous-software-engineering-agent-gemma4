"""Download competition data from Kaggle into ``data/competition``.

The competition rules forbid redistributing the data, so nothing under ``data/``
is tracked by git. This module fetches whichever groups a given workflow needs:

* ``core``       tasks.jsonl, harness README, sample submission, sandbox, docker (~2 MB)
* ``graphs``     pre-computed code graphs (~420 MB)
* ``embeddings`` pre-computed node embeddings (~470 MB)
* ``wheels``     offline wheels mounted into the sandbox (~28 MB)
* ``snapshots``  repository snapshots, one per task (~21.5 GB)
"""

from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMPETITION = "gemma-4-developer-agent"
WHEELHOUSE_DATASET = "metric/gemma-4-developer-agent-wheelhouse"
HARNESS_WHEEL_PREFIXES = ("swegemma-", "adk_submission-", "adk_eval_core-")

GROUPS = ("core", "graphs", "embeddings", "wheels", "snapshots")
_PREFIX_GROUPS = {
    "graphs/": "graphs",
    "embeddings/": "embeddings",
    "wheels/": "wheels",
    "snapshots/": "snapshots",
}


def group_of(name: str) -> str:
    """Map a competition file path to its download group."""
    for prefix, group in _PREFIX_GROUPS.items():
        if name.startswith(prefix):
            return group
    return "core"


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def list_files(api=None) -> list[tuple[str, int]]:
    """Return ``(name, size_bytes)`` for every file in the competition dataset."""
    api = api or _api()
    files: list[tuple[str, int]] = []
    token = None
    while True:
        page = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        files.extend((f.name, int(f.total_bytes or 0)) for f in page.files)
        token = page.next_page_token
        if not token:
            return files


def _fetch(api, name: str, size: int, dest_root: Path) -> str:
    target = dest_root / name
    if target.exists() and target.stat().st_size == size:
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    api.competition_download_file(COMPETITION, name, path=str(target.parent), quiet=True)
    # Kaggle zips some large files on download; unpack so the layout stays uniform.
    zipped = target.with_name(target.name + ".zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(target.parent)
        zipped.unlink()
    return "downloaded"


def download(
    groups: list[str],
    dest_root: Path,
    *,
    workers: int = 8,
    limit: int | None = None,
) -> dict[str, int]:
    """Download the requested groups, skipping files already present at full size."""
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown groups {unknown}; choose from {GROUPS}")
    api = _api()
    wanted = [(n, s) for n, s in list_files(api) if group_of(n) in groups]
    if limit is not None:
        wanted = wanted[:limit]
    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status in pool.map(lambda item: _fetch(api, item[0], item[1], dest_root), wanted):
            counts[status] += 1
    counts["bytes"] = sum(s for _, s in wanted)
    return counts


def download_harness_wheels(dest: Path) -> list[Path]:
    """Fetch the three pure-Python harness wheels from the official wheelhouse."""
    api = _api()
    dest.mkdir(parents=True, exist_ok=True)
    fetched: list[Path] = []
    token = None
    while True:
        page = api.dataset_list_files(WHEELHOUSE_DATASET, page_token=token, page_size=200)
        for f in page.files:
            if f.name.startswith(HARNESS_WHEEL_PREFIXES):
                api.dataset_download_file(WHEELHOUSE_DATASET, f.name, path=str(dest), quiet=True)
                fetched.append(dest / f.name)
        token = getattr(page, "next_page_token", None)
        if not token:
            return fetched
