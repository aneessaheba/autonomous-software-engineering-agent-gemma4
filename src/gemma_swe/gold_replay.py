"""Replay the reference patches through the official verification phase.

This answers a question a resolution rate cannot: how many public tasks can be
solved at all in the provided sandbox? Each task's reference ``patch`` is fed
to the harness as if the agent had produced it, so the usual Phase 2 runs
(fresh sandbox, apply patch, apply hidden tests, pytest). A task that fails
here is unsolvable locally whatever the agent does, typically because its
tests import a package the sandbox does not ship.

No model or GPU is involved, so this can run next to an evaluation.

    python -m gemma_swe.gold_replay --results-dir runs/gold-replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path


def run(args: argparse.Namespace) -> int:
    from adk_submission import ModelRegistry
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    try:
        from adk_eval_core.tracing import SessionTrace
    except ImportError:  # pragma: no cover - layout differs between harness versions
        from adk_eval_core.tracing.trace import SessionTrace

    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "task_results.jsonl"
    limits, constraints = build_submission_limits()
    config = EvalConfig(
        tasks_path=data_dir / "tasks.jsonl",
        snapshots_dir=data_dir / "snapshots",
        results_dir=results_dir,
        submission_dir=Path(args.agent_dir).resolve(),
        models=ModelRegistry(),
        sandbox=args.sandbox,
        verification_timeout_seconds=300,
        limits=limits,
        generation_constraints=constraints,
        graph_dir=str(data_dir / "graphs"),
        embeddings_dir=str(data_dir / "embeddings"),
        wheels_dir=data_dir / "wheels",
        display_mode="quiet",
        verbose=False,
    )
    evaluator = Evaluator(config)

    async def _gold_as_agent(task, *_args, **_kwargs):
        """Stand in for Phase 1: return the reference patch with no agent error."""
        return task.patch, None, SessionTrace()

    evaluator._run_agent_sandbox = _gold_as_agent

    tasks = load_tasks(data_dir / "tasks.jsonl")
    if args.task_ids:
        tasks = [t for t in tasks if t.instance_id in set(args.task_ids)]
    if args.limit:
        tasks = tasks[: args.limit]
    done = set()
    if args.resume and results_path.is_file():
        done = {json.loads(line)["instance_id"] for line in results_path.read_text().splitlines()}
    elif results_path.exists():
        results_path.unlink()
    tasks = [t for t in tasks if t.instance_id not in done]

    started = time.time()
    for idx, task in enumerate(tasks, start=1):
        try:
            result = asyncio.run(
                evaluator.evaluate_task(task=task, task_index=idx, total_tasks=len(tasks))
            )
            row = {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "resolved": bool(result.resolved),
                "test_exit_code": result.test_exit_code,
                "patch_chars": len(task.patch),
                "duration_seconds": round(result.duration_seconds or 0.0, 1),
                "error": getattr(result, "error_message", None),
            }
            output = getattr(result, "test_output", "") or ""
        except Exception as exc:  # keep going: one broken task must not hide the rest
            row = {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "resolved": False,
                "test_exit_code": None,
                "patch_chars": len(task.patch),
                "duration_seconds": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            output = ""
        with open(results_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        if not row["resolved"]:
            (results_dir / "test_outputs").mkdir(exist_ok=True)
            (results_dir / "test_outputs" / f"{task.instance_id}.log").write_text(output[-20000:])
        print(
            f"[{idx}/{len(tasks)}] {task.instance_id} gold_resolved={row['resolved']} "
            f"exit={row['test_exit_code']}",
            flush=True,
        )
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    solvable = sum(r["resolved"] for r in rows)
    print(
        f"Reference patch passes on {solvable} of {len(rows)} tasks "
        f"({(time.time() - started) / 60:.1f} min)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", default="runs/gold-replay")
    p.add_argument("--data-dir", default="data/competition")
    p.add_argument("--agent-dir", default="agents/baseline", help="Any valid agent directory")
    p.add_argument("--sandbox", choices=("subprocess", "docker"), default="subprocess")
    p.add_argument("--task-ids", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
