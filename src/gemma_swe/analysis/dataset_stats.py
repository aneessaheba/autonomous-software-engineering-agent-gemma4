"""Descriptive statistics for the public task set.

These numbers drive budget and prompt decisions: how big reference fixes are,
how many files they touch, and how much of the answer the issue text gives away.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import PurePosixPath

from gemma_swe.tasks import Task, is_test_path


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    s = sorted(values)

    def q(p: float) -> float:
        return float(s[min(len(s) - 1, int(p * len(s)))])

    return {
        "min": float(s[0]),
        "median": float(statistics.median(s)),
        "mean": round(statistics.fmean(s), 1),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": float(s[-1]),
    }


def mentions_path(task: Task, path: str) -> bool:
    """True if the issue text names the file by full path or by basename."""
    text = task.problem_statement
    return path in text or PurePosixPath(path).name in text


def mentions_module_stem(task: Task, path: str) -> bool:
    """True if the issue text contains the module stem as a whole word."""
    stem = PurePosixPath(path).stem
    if stem in {"__init__", "main", "utils", "base"} or len(stem) < 4:
        return False
    return (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(stem)}(?![A-Za-z0-9_])", task.problem_statement)
        is not None
    )


def compute_stats(tasks: list[Task]) -> dict:
    n = len(tasks)
    src_file_counts: list[int] = []
    added: list[int] = []
    removed: list[int] = []
    statement_chars: list[int] = []
    new_file_tasks = non_py_tasks = single_file = path_named = stem_named = 0
    test_only = 0
    ext_counter: Counter[str] = Counter()
    top_dirs: Counter[str] = Counter()

    for t in tasks:
        src = [f for f in t.gold_files if not is_test_path(f.path)]
        src_file_counts.append(len(src))
        added.append(sum(f.added for f in src))
        removed.append(sum(f.removed for f in src))
        statement_chars.append(len(t.problem_statement))
        if not src:
            test_only += 1
            continue
        if len(src) == 1:
            single_file += 1
        if any(f.is_new for f in src):
            new_file_tasks += 1
        if any(not f.path.endswith(".py") for f in src):
            non_py_tasks += 1
        if any(mentions_path(t, f.path) for f in src):
            path_named += 1
        if any(mentions_module_stem(t, f.path) for f in src):
            stem_named += 1
        for f in src:
            ext_counter[PurePosixPath(f.path).suffix or "(none)"] += 1
            top_dirs[f"{t.repo_short}/{PurePosixPath(f.path).parts[0]}"] += 1

    size_buckets = Counter()
    for a, r in zip(added, removed, strict=True):
        changed = a + r
        bucket = (
            "1-5"
            if changed <= 5
            else "6-20"
            if changed <= 20
            else "21-100"
            if changed <= 100
            else ">100"
        )
        size_buckets[bucket] += 1

    return {
        "n_tasks": n,
        "tasks_per_repo": dict(Counter(t.repo for t in tasks).most_common()),
        "tasks_per_year": dict(sorted(Counter(t.created_at[:4] for t in tasks).items())),
        "with_hints": sum(1 for t in tasks if t.hints_text.strip()),
        "problem_statement_chars": _quantiles(statement_chars),
        "gold_source_files_per_task": _quantiles(src_file_counts),
        "gold_lines_added": _quantiles(added),
        "gold_lines_removed": _quantiles(removed),
        "changed_lines_buckets": {k: size_buckets[k] for k in ("1-5", "6-20", "21-100", ">100")},
        "single_source_file_pct": _pct(single_file, n),
        "creates_new_file_pct": _pct(new_file_tasks, n),
        "touches_non_python_pct": _pct(non_py_tasks, n),
        "issue_names_gold_file_pct": _pct(path_named, n),
        "issue_names_gold_module_stem_pct": _pct(stem_named, n),
        "tasks_without_source_change": test_only,
        "gold_file_extensions": dict(ext_counter.most_common(8)),
        "gold_top_level_dirs": dict(top_dirs.most_common(12)),
    }


def to_markdown(stats: dict) -> str:
    def row(label: str, q: dict) -> str:
        return f"| {label} | {q['median']:g} | {q['mean']:g} | {q['p75']:g} | {q['p90']:g} | {q['max']:g} |"

    lines = [
        "# Public task set: descriptive statistics",
        "",
        f"Computed over the {stats['n_tasks']} public development tasks in `tasks.jsonl`.",
        "Regenerate with `gemma-swe stats`.",
        "",
        "## Composition",
        "",
        "| Repository | Tasks |",
        "| :--- | ---: |",
        *[f"| {r} | {c} |" for r, c in stats["tasks_per_repo"].items()],
        "",
        f"Tasks with maintainer hints: {stats['with_hints']} of {stats['n_tasks']}.",
        "",
        "## Size of the reference fix (non-test files only)",
        "",
        "| Quantity | Median | Mean | P75 | P90 | Max |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
        row("Source files touched", stats["gold_source_files_per_task"]),
        row("Lines added", stats["gold_lines_added"]),
        row("Lines removed", stats["gold_lines_removed"]),
        row("Issue length (chars)", stats["problem_statement_chars"]),
        "",
        "| Changed lines (added + removed) | Tasks |",
        "| :--- | ---: |",
        *[f"| {k} | {v} |" for k, v in stats["changed_lines_buckets"].items()],
        "",
        "## What the issue text gives away",
        "",
        "| Property | Share of tasks |",
        "| :--- | ---: |",
        f"| Fix is confined to one source file | {stats['single_source_file_pct']}% |",
        f"| Fix creates at least one new file | {stats['creates_new_file_pct']}% |",
        f"| Fix touches a non-Python file | {stats['touches_non_python_pct']}% |",
        f"| Issue names a gold file (path or basename) | {stats['issue_names_gold_file_pct']}% |",
        f"| Issue names a gold module stem as a word | {stats['issue_names_gold_module_stem_pct']}% |",
        "",
        "## Where fixes land",
        "",
        "| Top-level directory | Gold files |",
        "| :--- | ---: |",
        *[f"| `{k}` | {v} |" for k, v in stats["gold_top_level_dirs"].items()],
        "",
    ]
    return "\n".join(lines)
