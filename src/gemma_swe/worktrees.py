"""Extract lightweight working trees from the competition's repository snapshots.

A snapshot is a ~200 MB tarball holding the working tree plus its git history.
Offline analysis only needs the Python sources at ``base_commit``, so this
extracts ``*.py`` files and nothing else (a few MB per task).
"""

from __future__ import annotations

import tarfile
from pathlib import Path

SNAPSHOT_DIR = Path("data/competition/snapshots")
WORKTREE_DIR = Path("data/worktrees")


def extract_python_tree(instance_id: str, *, force: bool = False) -> Path | None:
    """Extract the task's ``*.py`` files; return the tree root or ``None`` if unavailable."""
    archive = SNAPSHOT_DIR / f"{instance_id}.tgz"
    dest = WORKTREE_DIR / instance_id
    marker = dest / ".extracted"
    if marker.exists() and not force:
        return dest
    if not archive.is_file():
        return None
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = [
                m
                for m in tar
                if m.isfile() and m.name.endswith(".py") and "/.git/" not in f"/{m.name}/"
            ]
            tar.extractall(dest, members=members, filter="data")
    except (tarfile.TarError, EOFError, OSError):
        return None  # still downloading, or truncated
    marker.write_text(f"{len(members)}\n")
    return dest
