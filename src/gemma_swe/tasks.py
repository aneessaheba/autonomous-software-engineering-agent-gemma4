"""Task loading and unified-diff parsing for the competition's ``tasks.jsonl``."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DEFAULT_TASKS_PATH = Path("data/competition/tasks.jsonl")

_FILE_HEADER = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.MULTILINE)
_OLD_HEADER = re.compile(r"^--- (?:a/)?(\S+)", re.MULTILINE)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_TEST_DIRS = {"tests", "test", "testing"}


def is_test_path(path: str) -> bool:
    """Mirror the harness definition of a protected test file."""
    p = PurePosixPath(path)
    if p.suffix != ".py":
        return False
    if p.name.startswith("test_") or p.name.endswith("_test.py"):
        return True
    return any(part in _TEST_DIRS for part in p.parts[:-1])


@dataclass(frozen=True)
class FileDiff:
    """One file's worth of a unified diff."""

    path: str
    added: int
    removed: int
    is_new: bool
    # 1-indexed inclusive line ranges in the *pre-patch* file touched by each hunk.
    old_ranges: tuple[tuple[int, int], ...] = ()


def parse_patch(patch: str) -> list[FileDiff]:
    """Split a unified diff into per-file summaries.

    Handles both ``diff --git`` style patches and the bare ``---``/``+++`` form
    used in the competition's ``patch`` and ``test_patch`` fields.
    """
    diffs: list[FileDiff] = []
    # Split at each "--- " header that is immediately followed by a "+++ " header.
    starts = [m.start() for m in re.finditer(r"^--- \S+.*\n\+\+\+ \S+", patch, flags=re.MULTILINE)]
    for i, start in enumerate(starts):
        chunk = patch[start : starts[i + 1] if i + 1 < len(starts) else len(patch)]
        new = _FILE_HEADER.search(chunk)
        old = _OLD_HEADER.search(chunk)
        if not new:
            continue
        path = new.group(1)
        is_new = bool(old and old.group(1) == "/dev/null")
        if path == "/dev/null" and old:  # deletion
            path = old.group(1)
        added = removed = 0
        ranges: list[tuple[int, int]] = []
        for line in chunk.split("\n")[2:]:
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        for h in _HUNK.finditer(chunk):
            old_start = int(h.group(1))
            old_len = int(h.group(2)) if h.group(2) is not None else 1
            if old_start == 0 and old_len == 0:
                is_new = True
                continue
            ranges.append((old_start, old_start + max(old_len, 1) - 1))
        diffs.append(FileDiff(path, added, removed, is_new, tuple(ranges)))
    return diffs


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    patch: str = ""
    test_patch: str = ""
    created_at: str = ""
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def repo_short(self) -> str:
        return self.repo.split("/")[-1]

    @property
    def gold_files(self) -> list[FileDiff]:
        if "gold" not in self._cache:
            self._cache["gold"] = parse_patch(self.patch)
        return self._cache["gold"]

    @property
    def test_files(self) -> list[FileDiff]:
        if "test" not in self._cache:
            self._cache["test"] = parse_patch(self.test_patch)
        return self._cache["test"]

    @property
    def gold_source_paths(self) -> list[str]:
        """Non-test files the reference fix touches: what an agent must find and edit."""
        return [f.path for f in self.gold_files if not is_test_path(f.path)]

    def graph_stem(self) -> str:
        """File stem of this task's commit-named graph / embedding artefact."""
        return f"{self.repo_short}_{self.base_commit}"


def load_tasks(path: str | Path = DEFAULT_TASKS_PATH) -> list[Task]:
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks.append(
                Task(
                    instance_id=row["instance_id"],
                    repo=row["repo"],
                    base_commit=row["base_commit"],
                    problem_statement=row.get("problem_statement") or "",
                    hints_text=row.get("hints_text") or "",
                    patch=row.get("patch") or "",
                    test_patch=row.get("test_patch") or "",
                    created_at=row.get("created_at") or "",
                )
            )
    return tasks
