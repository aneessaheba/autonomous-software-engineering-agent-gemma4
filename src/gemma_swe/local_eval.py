"""Run the official two-phase evaluation locally on a GPU machine.

This mirrors the organisers' "Getting Started" notebook: it serves the declared
Gemma 4 model with vLLM through ``adk_submission.VllmServer``, then drives
``swegemma.evaluate.Evaluator`` with the same context-compaction and caching
settings the scorer uses, so a local resolution rate is comparable to the
leaderboard. It needs a CUDA GPU; see ``scripts/gpu/setup.sh``.

Each task appends one line to ``<results-dir>/task_results.jsonl`` as soon as it
finishes, so an interrupted run can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ENV_DEFAULTS = {
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_FLAX": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_TELEMETRY": "False",
    "LITELLM_LOG": "WARNING",
    "OPENAI_API_KEY": "EMPTY",
    "LOCAL_API_KEY": "EMPTY",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "OTEL_SDK_DISABLED": "true",
}
MODEL_VLLM_EXTRA_ARGS = {
    "gemma-4-12b-it-qat-w4a16-ct": (
        "--hf-overrides",
        '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
    ),
}


def _load_eval_section(agent_dir: Path) -> dict:
    import yaml

    for name in ("eval_config.yaml", "eval_config.yml"):
        path = agent_dir / name
        if path.is_file():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return raw.get("evaluation", raw)
    return {}


def _done_ids(results_path: Path) -> set[str]:
    if not results_path.is_file():
        return set()
    return {
        json.loads(line)["instance_id"] for line in results_path.read_text().splitlines() if line
    }


def run(args: argparse.Namespace) -> int:
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from adk_submission.discovery import discover_declared_models
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    litellm.drop_params = True
    agent_dir = Path(args.agent_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "task_results.jsonl"

    declared = discover_declared_models(agent_dir)
    if len(declared) != 1:
        raise SystemExit(f"Expected exactly one declared model, got {sorted(declared)}")
    model_name = next(iter(declared))
    model_path = Path(args.model_dir) / model_name
    if not model_path.exists():
        raise SystemExit(f"Model weights not found at {model_path}")

    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, "adapters", None) or {}
    ranks = [r for info in adapter_map.values() if (r := getattr(info, "rank", None))]
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tp_size = 4 if gpus >= 4 else (2 if gpus >= 2 else 1)

    server = VllmServer(
        VllmConfig(
            model=str(model_path),
            port=args.port,
            host="127.0.0.1",
            tool_call_parser="gemma4",
            reasoning_parser="gemma4",
            max_model_len=32768,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_auto_tool_choice=True,
            enable_lora=bool(adapter_map),
            max_loras=max(1, len(adapter_map)),
            max_lora_rank=max(ranks) if ranks else 128,
            scheduler_reserve_full_isl=False,
            tensor_parallel_size=tp_size,
            startup_timeout=60 * 20,
            extra_args=list(MODEL_VLLM_EXTRA_ARGS.get(model_name, ())),
        ),
        adapter_manifest=adapters,
    )
    server.start()
    try:
        models = server.create_model_registry(
            aliases=[model_name], model_prefix="openai/", api_key=os.environ["LOCAL_API_KEY"]
        )
        section = _load_eval_section(agent_dir)
        turns = section.get("max_turns", section.get("max_llm_calls"))
        limits, constraints = build_submission_limits()
        config = EvalConfig(
            tasks_path=data_dir / "tasks.jsonl",
            snapshots_dir=data_dir / "snapshots",
            results_dir=results_dir,
            submission_dir=agent_dir,
            models=models,
            sandbox=args.sandbox,
            timeout_seconds=int(section.get("timeout_seconds", 300)),
            verification_timeout_seconds=int(section.get("verification_timeout_seconds", 300)),
            max_time_minutes=float(section.get("max_time_minutes", 60.0)),
            max_tool_calls=int(section.get("max_tool_calls", 100)),
            max_turns=int(turns) if turns is not None else None,
            limits=limits,
            generation_constraints=constraints,
            adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(
                min_tokens=2048, ttl_seconds=1800, cache_intervals=10
            ),
            events_compaction_config=EventsCompactionConfig(
                compaction_interval=15,
                overlap_size=2,
                token_threshold=14336,
                event_retention_size=5,
            ),
            graph_dir=str(data_dir / "graphs"),
            embeddings_dir=str(data_dir / "embeddings"),
            wheels_dir=data_dir / "wheels",
            verbose=False,
        )
        evaluator = Evaluator(config)

        tasks = load_tasks(data_dir / "tasks.jsonl")
        if args.task_ids:
            wanted = set(args.task_ids)
            tasks = [t for t in tasks if t.instance_id in wanted]
        if args.repo:
            tasks = [t for t in tasks if re.search(args.repo, t.repo)]
        if args.num_shards:
            tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard_index]
        if args.limit:
            tasks = tasks[: args.limit]
        done = _done_ids(results_path) if args.resume else set()
        if not args.resume and results_path.exists():
            results_path.unlink()
        tasks = [t for t in tasks if t.instance_id not in done]

        started = time.time()
        for idx, task in enumerate(tasks, start=1):
            result = asyncio.run(
                evaluator.evaluate_task(task=task, task_index=idx, total_tasks=len(tasks))
            )
            patch = (result.agent_patch or "").strip()
            row = {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "resolved": bool(result.resolved),
                "test_exit_code": result.test_exit_code,
                "patch_chars": len(patch),
                "tool_calls": result.tool_calls,
                "duration_seconds": round(result.duration_seconds or 0.0, 1),
                "status": str(getattr(result, "status", "")),
                "total_tokens": getattr(result, "total_tokens", None),
                "llm_calls": getattr(result, "total_llm_calls", None),
                "error": getattr(result, "error_message", None),
            }
            with open(results_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            (results_dir / "patches").mkdir(exist_ok=True)
            (results_dir / "patches" / f"{task.instance_id}.patch").write_text(patch)
            print(
                f"[{idx}/{len(tasks)}] {task.instance_id} resolved={row['resolved']} "
                f"calls={row['tool_calls']} {row['duration_seconds']}s",
                flush=True,
            )
        print(f"Finished {len(tasks)} tasks in {(time.time() - started) / 60:.1f} min")
    finally:
        server.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("agent_dir", help="Agent variant directory, e.g. agents/locate")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--data-dir", default="data/competition")
    p.add_argument("--model-dir", default="data/models", help="Holds one folder per model alias")
    p.add_argument("--sandbox", choices=("subprocess", "docker"), default="subprocess")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--task-ids", nargs="*")
    p.add_argument("--repo", help="Regex filter on owner/repo")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int)
    p.add_argument("--resume", action="store_true", help="Skip tasks already in task_results.jsonl")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
