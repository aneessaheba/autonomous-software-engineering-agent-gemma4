"""Offline evaluation of fault localization on the public tasks.

Question: given only the issue text, how often does a ranking method put a file
the reference fix edits near the top? The answer bounds how much exploration
budget the agent needs, and tells us whether shipping a localization script as
a skill is worth a tool call.

Three methods are compared on identical inputs (the task's ``*.py`` files at
``base_commit``, tests excluded):

* ``ident_grep``  what a careful agent does by hand with ``git grep``: take the
                  code identifiers from the issue and rank files by the
                  IDF-weighted number of distinct identifiers they contain.
* ``bm25_file``   BM25 with one document per file and the whole issue as query.
* ``locate``      the script shipped in ``agents/locate/skills/repo-locate``:
                  BM25 over function/class chunks with symbol and path boosts.

Gold labels come from the reference ``patch``. Files the patch creates cannot
be localized, so tasks whose fix only adds new files are reported separately.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gemma_swe.tasks import Task, is_test_path
from gemma_swe.worktrees import extract_python_tree

LOCATE_SCRIPT = Path("agents/locate/skills/repo-locate/scripts/locate.py")
KS = (1, 3, 5, 10)
METHODS = ("ident_grep", "bm25_file", "locate")


def load_locate(script: Path = LOCATE_SCRIPT):
    spec = importlib.util.spec_from_file_location("repo_locate", script)
    module = importlib.util.module_from_spec(spec)
    # Importing must not leave a __pycache__ inside the submission directory.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _read_files(locate, root: str) -> dict[str, str]:
    files = {}
    for path in locate.list_python_files(root):
        try:
            with open(os.path.join(root, path), encoding="utf-8", errors="replace") as fh:
                files[path] = fh.read()
        except OSError:
            continue
    return files


def rank_ident_grep(locate, files: dict[str, str], query: str) -> list[str]:
    idents = {i.lower() for i in locate.code_identifiers(query)}
    if not idents:
        return []
    present = {p: idents & set(locate.tokenize(text)) for p, text in files.items()}
    df = Counter(tok for toks in present.values() for tok in toks)
    n = len(files)
    scores = {
        p: sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in toks)
        for p, toks in present.items()
        if toks
    }
    return sorted(scores, key=lambda p: (-scores[p], p))


def rank_bm25_file(locate, files: dict[str, str], query: str, k1=1.2, b=0.75) -> list[str]:
    tfs = {p: Counter(locate.tokenize(text)) for p, text in files.items()}
    lengths = {p: sum(tf.values()) for p, tf in tfs.items()}
    n = len(tfs)
    avg = (sum(lengths.values()) / n) if n else 1.0
    df = Counter(tok for tf in tfs.values() for tok in tf)
    q = [t for t in set(locate.tokenize(query)) if t in df]
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q}
    scores = {}
    for p, tf in tfs.items():
        norm = k1 * (1 - b + b * lengths[p] / avg)
        s = sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        if s > 0:
            scores[p] = s
    return sorted(scores, key=lambda p: (-scores[p], p))


def _symbol_hit(symbols, ranges) -> bool:
    return any(s <= hi and lo <= e for _, s, e, _ in symbols for lo, hi in ranges)


def evaluate_task(task: Task) -> dict | None:
    root = extract_python_tree(task.instance_id)
    if root is None:
        return None
    locate = load_locate()
    files = _read_files(locate, str(root))
    gold = [f for f in task.gold_files if not is_test_path(f.path)]
    existing = {f.path for f in gold if f.path in files}
    row = {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "n_candidate_files": len(files),
        "n_gold_files": len(gold),
        "n_gold_existing": len(existing),
        "localizable": bool(existing),
    }
    if not existing:
        return row

    index = locate.Index(str(root))
    located = index.rank(task.problem_statement, top_files=max(KS), per_file=3)
    rankings = {
        "ident_grep": rank_ident_grep(locate, files, task.problem_statement),
        "bm25_file": rank_bm25_file(locate, files, task.problem_statement),
        "locate": [path for path, _, _ in located],
    }
    for method, ranking in rankings.items():
        first = next((i for i, p in enumerate(ranking, 1) if p in existing), None)
        row[f"{method}_first_rank"] = first
        for k in KS:
            top = set(ranking[:k])
            row[f"{method}_hit@{k}"] = bool(top & existing)
            row[f"{method}_recall@{k}"] = len(top & existing) / len(existing)

    # Function level: does a suggested line range overlap a line the fix changes?
    gold_ranges = {f.path: f.old_ranges for f in gold if f.path in existing and f.old_ranges}
    if gold_ranges:
        for k in (1, 3, 5):
            row[f"locate_symbol_hit@{k}"] = any(
                path in gold_ranges and _symbol_hit(symbols, gold_ranges[path])
                for path, _, symbols in located[:k]
            )
    return row


def evaluate(tasks: list[Task], workers: int = 8) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(evaluate_task, tasks, chunksize=2))
    return [r for r in rows if r is not None]


def summarise(rows: list[dict]) -> dict:
    loc = [r for r in rows if r["localizable"]]
    n = len(loc)

    def mean(key: str, subset=None) -> float | None:
        vals = [float(r[key]) for r in (subset or loc) if key in r]
        return round(100 * sum(vals) / len(vals), 1) if vals else None

    def mrr(method: str, subset=None) -> float:
        subset = subset or loc
        ranks = [r[f"{method}_first_rank"] for r in subset]
        return round(sum(1 / x for x in ranks if x) / len(ranks), 3) if ranks else 0.0

    summary = {
        "n_tasks_evaluated": len(rows),
        "n_localizable": n,
        "n_new_files_only": len(rows) - n,
        "median_candidate_files": sorted(r["n_candidate_files"] for r in rows)[len(rows) // 2]
        if rows
        else 0,
        "methods": {},
        "by_repo": {},
    }
    for method in METHODS:
        summary["methods"][method] = {
            **{f"hit@{k}": mean(f"{method}_hit@{k}") for k in KS},
            **{f"recall@{k}": mean(f"{method}_recall@{k}") for k in (5, 10)},
            "mrr": mrr(method),
        }
    summary["locate_symbol"] = {f"hit@{k}": mean(f"locate_symbol_hit@{k}") for k in (1, 3, 5)} | {
        "n": sum(1 for r in loc if "locate_symbol_hit@1" in r)
    }
    for repo in sorted({r["repo"] for r in loc}):
        sub = [r for r in loc if r["repo"] == repo]
        summary["by_repo"][repo] = {
            "n": len(sub),
            **{
                m: {"hit@1": mean(f"{m}_hit@1", sub), "hit@5": mean(f"{m}_hit@5", sub)}
                for m in METHODS
            },
        }
    return summary


def to_markdown(summary: dict) -> str:
    labels = {
        "ident_grep": "Identifier grep (manual `git grep` strategy)",
        "bm25_file": "BM25, one document per file",
        "locate": "`repo-locate` skill (chunk BM25 + boosts)",
    }
    lines = [
        "# Fault localization from issue text",
        "",
        f"Evaluated on {summary['n_tasks_evaluated']} public tasks; "
        f"{summary['n_localizable']} have a reference fix that edits an existing file, "
        f"{summary['n_new_files_only']} only create new files and are excluded below.",
        f"Median candidate pool: {summary['median_candidate_files']} non-test Python files per task.",
        "Regenerate with `gemma-swe localize`.",
        "",
        "## File level: is a file the reference fix edits in the top k?",
        "",
        "| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        m = summary["methods"][method]
        lines.append(
            f"| {labels[method]} | {m['hit@1']}% | {m['hit@3']}% | {m['hit@5']}% | "
            f"{m['hit@10']}% | {m['recall@5']}% | {m['recall@10']}% | {m['mrr']} |"
        )
    sym = summary["locate_symbol"]
    lines += [
        "",
        "Hit@k: at least one gold file in the top k. Recall@k: share of gold files in the top k.",
        "",
        "## Function level",
        "",
        f"For the {sym['n']} tasks whose fix modifies existing lines: does one of the three line "
        "ranges the skill prints for a top-k file overlap a line the reference fix changes?",
        "",
        "| Top-k files | Hit rate |",
        "| :--- | ---: |",
        *[f"| {k} | {sym[f'hit@{k}']}% |" for k in (1, 3, 5)],
        "",
        "## By repository (file level)",
        "",
        "| Repository | Tasks | Method | Hit@1 | Hit@5 |",
        "| :--- | ---: | :--- | ---: | ---: |",
    ]
    for repo, data in summary["by_repo"].items():
        for method in METHODS:
            lines.append(
                f"| {repo} | {data['n']} | {method} | {data[method]['hit@1']}% | {data[method]['hit@5']}% |"
            )
    lines.append("")
    return "\n".join(lines)
