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


EXIT_LABELS = {
    "0": "tests passed",
    "1": "tests ran and failed",
    "2": "tests could not be collected (import or syntax error)",
    "None": "tests not run",
}


def load_solvable(gold_dir: str | Path) -> set[str]:
    """Task ids whose reference patch passes verification (from ``gold_replay``)."""
    return {r["instance_id"] for r in load_rows(gold_dir) if r.get("resolved")}


def summarise(rows: list[dict], solvable: set[str] | None = None) -> dict:
    n = len(rows)
    resolved = sum(1 for r in rows if r.get("resolved"))
    lo, hi = wilson_interval(resolved, n)
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r.get("repo", "?")].append(r)
    durations = [r["duration_seconds"] for r in rows if r.get("duration_seconds")]
    calls = [r["tool_calls"] for r in rows if r.get("tool_calls") is not None]
    exit_codes: dict[str, int] = defaultdict(int)
    for r in rows:
        exit_codes[str(r.get("test_exit_code"))] += 1
    on_solvable = None
    if solvable is not None:
        sub = [r for r in rows if r["instance_id"] in solvable]
        k = sum(1 for r in sub if r.get("resolved"))
        s_lo, s_hi = wilson_interval(k, len(sub))
        on_solvable = {
            "n_tasks": len(sub),
            "resolved": k,
            "resolution_rate": round(k / len(sub), 4) if sub else 0.0,
            "ci95": [round(s_lo, 4), round(s_hi, 4)],
        }
    tokens = [r["total_tokens"] for r in rows if r.get("total_tokens")]
    return {
        "n_tasks": n,
        "resolved": resolved,
        "on_solvable_tasks": on_solvable,
        "test_exit_codes": dict(sorted(exit_codes.items())),
        "timeouts": sum(1 for r in rows if "timeout" in str(r.get("error") or "").lower()),
        "median_total_tokens": statistics.median(tokens) if tokens else None,
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
        *(
            [
                "",
                f"On the {summary['on_solvable_tasks']['n_tasks']} tasks whose reference patch "
                f"passes in this sandbox: {summary['on_solvable_tasks']['resolved']} resolved "
                f"({100 * summary['on_solvable_tasks']['resolution_rate']:.1f}%, 95% CI "
                f"{100 * summary['on_solvable_tasks']['ci95'][0]:.1f} to "
                f"{100 * summary['on_solvable_tasks']['ci95'][1]:.1f}%).",
            ]
            if summary.get("on_solvable_tasks")
            else []
        ),
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Empty patches | {summary['empty_patches']} |",
        f"| Timeouts | {summary['timeouts']} |",
        f"| Median total tokens per task | {summary['median_total_tokens']} |",
        f"| Harness errors | {summary['errors']} |",
        f"| Median tool calls | {summary['median_tool_calls']} |",
        f"| Median task time (s) | {summary['median_duration_seconds']} |",
        f"| Total agent time (h) | {summary['total_hours']} |",
        "",
        "| Test outcome | Tasks |",
        "| :--- | ---: |",
        *[
            f"| exit {code}: {EXIT_LABELS.get(code, 'runner error or kill')} | {count} |"
            for code, count in summary["test_exit_codes"].items()
        ],
        "",
        "| Repository | Tasks | Resolved |",
        "| :--- | ---: | ---: |",
        *[f"| {repo} | {d['n']} | {d['resolved']} |" for repo, d in summary["by_repo"].items()],
        "",
    ]
    return "\n".join(lines)
