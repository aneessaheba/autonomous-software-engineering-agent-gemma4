"""Command line entry point: ``gemma-swe <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path("data/competition")
RESULTS_DIR = Path("results")


def _cmd_download(args: argparse.Namespace) -> int:
    from gemma_swe import kaggle_data

    if args.harness_wheels:
        wheels = kaggle_data.download_harness_wheels(Path("vendor/wheels"))
        print("harness wheels:", ", ".join(w.name for w in wheels))
    if args.groups:
        counts = kaggle_data.download(args.groups, DATA_DIR, workers=args.workers)
        print(json.dumps(counts))
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from gemma_swe.analysis.dataset_stats import compute_stats, to_markdown
    from gemma_swe.tasks import load_tasks

    stats = compute_stats(load_tasks(args.tasks))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (RESULTS_DIR / "dataset_stats.md").write_text(to_markdown(stats))
    print(to_markdown(stats))
    return 0


def _cmd_localize(args: argparse.Namespace) -> int:
    from gemma_swe.analysis.localization import evaluate, summarise, to_markdown
    from gemma_swe.tasks import load_tasks

    rows = evaluate(load_tasks(args.tasks), workers=args.workers)
    summary = summarise(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "localization.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "localization.md").write_text(to_markdown(summary))
    if args.per_task:
        with open(args.per_task, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    print(to_markdown(summary))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from gemma_swe.submission import validate

    report = validate(args.submission_dir, use_harness=not args.no_harness)
    for key, value in report.info.items():
        print(f"{key}: {value}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("VALID" if report.ok else "INVALID")
    return 0 if report.ok else 1


def _cmd_package(args: argparse.Namespace) -> int:
    from gemma_swe.submission import package, validate

    report = validate(args.submission_dir, use_harness=not args.no_harness)
    if not report.ok:
        for error in report.errors:
            print(f"ERROR: {error}")
        print("Refusing to package an invalid submission.")
        return 1
    manifest = package(args.submission_dir, args.out)
    manifest["validation"] = report.info
    Path(args.out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-swe", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download competition data from Kaggle")
    p.add_argument("groups", nargs="*", help="core graphs embeddings wheels snapshots")
    p.add_argument("--harness-wheels", action="store_true", help="Also fetch harness wheels")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("stats", help="Descriptive statistics for the public tasks")
    p.add_argument("--tasks", default=str(DATA_DIR / "tasks.jsonl"))
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("localize", help="Evaluate fault localization on the public tasks")
    p.add_argument("--tasks", default=str(DATA_DIR / "tasks.jsonl"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-task", help="Also write per-task rows to this JSONL file")
    p.set_defaults(func=_cmd_localize)

    p = sub.add_parser("validate", help="Validate a submission directory")
    p.add_argument("submission_dir", nargs="?", default="agents/locate")
    p.add_argument("--no-harness", action="store_true", help="Skip official harness checks")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("package", help="Validate, then write submission.zip")
    p.add_argument("submission_dir", nargs="?", default="agents/locate")
    p.add_argument("--out", default="dist/submission.zip")
    p.add_argument("--no-harness", action="store_true")
    p.set_defaults(func=_cmd_package)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
