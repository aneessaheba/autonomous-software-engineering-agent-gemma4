"""Summarise a local evaluation run and compare runs task by task."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(results_dir: str | Path) -> list[dict]:
    path = Path(results_dir) / "task_results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; honest for small n."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved"))
    lo, hi = wilson_interval(resolved, n)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r.get("repo", "?")].append(r)
    durations = [r["duration_seconds"] for r in rows if r.get("duration_seconds")]
    calls = [r["tool_calls"] for r in rows if r.get("tool_calls") is not None]
    return {
        "n_tasks": n,
        "resolved": resolved,
        "resolution_rate": round(resolved / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "empty_patches": sum(1 for r in rows if not r.get("patch_chars")),
        "errors": sum(1 for r in rows if r.get("error")),
        "median_tool_calls": statistics.median(calls) if calls else None,
        "median_duration_seconds": statistics.median(durations) if durations else None,
        "total_hours": round(sum(durations) / 3600, 2) if durations else None,
        "by_repo": {
            repo: {"n": len(rs), "resolved": sum(1 for r in rs if r.get("resolved"))}
            for repo, rs in sorted(by_repo.items())
        },
    }


def compare(base: list[dict], other: list[dict]) -> dict:
    """Paired comparison on the tasks both runs attempted."""
    a = {r["instance_id"]: bool(r.get("resolved")) for r in base}
    b = {r["instance_id"]: bool(r.get("resolved")) for r in other}
    shared = sorted(a.keys() & b.keys())
    gained = [i for i in shared if b[i] and not a[i]]
    lost = [i for i in shared if a[i] and not b[i]]
    return {
        "n_shared": len(shared),
        "base_resolved": sum(a[i] for i in shared),
        "other_resolved": sum(b[i] for i in shared),
        "gained": gained,
        "lost": lost,
        "net": len(gained) - len(lost),
    }


def to_markdown(name: str, summary: dict) -> str:
    lo, hi = summary["ci95"]
    lines = [
        f"## {name}",
        "",
        f"Resolved {summary['resolved']} of {summary['n_tasks']} "
        f"({100 * summary['resolution_rate']:.1f}%, 95% CI {100 * lo:.1f} to {100 * hi:.1f}%).",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Empty patches | {summary['empty_patches']} |",
        f"| Harness errors | {summary['errors']} |",
        f"| Median tool calls | {summary['median_tool_calls']} |",
        f"| Median task time (s) | {summary['median_duration_seconds']} |",
        f"| Total agent time (h) | {summary['total_hours']} |",
        "",
        "| Repository | Tasks | Resolved |",
        "| :--- | ---: | ---: |",
        *[f"| {repo} | {d['n']} | {d['resolved']} |" for repo, d in summary["by_repo"].items()],
        "",
    ]
    return "\n".join(lines)
