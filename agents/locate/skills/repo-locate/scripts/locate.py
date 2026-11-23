"""Rank the files and functions most likely to need editing for an issue.

Runs inside the offline task sandbox, so it uses the standard library only.
It splits every Python source file into function, method and class chunks,
scores each chunk against the query with BM25 over code-aware tokens, and
prints the best files with the line ranges worth reading first.

Usage:
    locate.py "issue keywords, identifiers, error messages"
    locate.py --top 8 --root /workspace "query"
"""

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter

SKIP_DIRS = {
    ".git", ".hg", ".tox", ".venv", "venv", "env", "node_modules", "build", "dist",
    "site-packages", "__pycache__", ".eggs", ".mypy_cache", ".pytest_cache",
}  # fmt: skip
TEST_DIRS = {"tests", "test", "testing"}
MAX_FILE_BYTES = 600_000
K1, B = 1.2, 0.75

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may me might more most must my no not of on or our should so some such than
    that the their them then there these they this to too use used using was we were what when
    where which while who why will with would you your also any all each just like only other
    over same very after before about above below between both because being here
    def class return self cls none true false import pass else elif try except finally raise
    lambda yield async await assert del global nonlocal while break continue print
    str int bool float list dict tuple set type object args kwargs len range isinstance
    py python code file files line lines error errors issue bug fix fixed fixes example
    expected actual behavior behaviour version description steps reproduce following
    get new value values name names data result make add added adds support test tests
    """.split()
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_CODE_SPAN = re.compile(r"`([^`\n]{2,80})`")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def subtokens(identifier):
    """Split ``get_HTTPResponse`` into ``get``, ``http``, ``response``."""
    parts = []
    for piece in identifier.split("_"):
        parts.extend(m.group().lower() for m in _CAMEL.finditer(piece))
    return [p for p in parts if len(p) > 1]


def tokenize(text):
    """Emit each identifier whole (lowercased) plus its snake/camel sub-tokens."""
    tokens = []
    for ident in _IDENT.findall(text):
        low = ident.lower()
        if len(low) < 2 or low in STOPWORDS:
            continue
        tokens.append(low)
        subs = subtokens(ident)
        if len(subs) > 1:
            tokens.extend(s for s in subs if s not in STOPWORDS)
    return tokens


def code_identifiers(query):
    """Identifiers the query marks as code: backticked, dotted, snake or camel case."""
    found = set()
    for span in _CODE_SPAN.findall(query):
        found.update(_IDENT.findall(span))
    for dotted in _DOTTED.findall(query):
        found.update(dotted.split("."))
    for ident in _IDENT.findall(query):
        if "_" in ident.strip("_") or re.search(r"[a-z][A-Z]", ident):
            found.add(ident)
    return {i for i in found if len(i) > 2 and i.lower() not in STOPWORDS}


def is_test_path(path):
    parts = path.split("/")
    name = parts[-1]
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return True
    return any(p in TEST_DIRS for p in parts[:-1])


def list_python_files(root, include_tests=False):
    paths = []
    try:
        # Only trust git when ``root`` is itself the repository top level; otherwise
        # an enclosing repository's ignore rules would hide every file.
        if not os.path.exists(os.path.join(root, ".git")):
            raise LookupError("not a git work tree root")
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.py"],
            cwd=root, capture_output=True, timeout=30, check=True,
        ).stdout.decode("utf-8", "replace")  # fmt: skip
        paths = [p for p in out.split("\0") if p]
    except Exception:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                if name.endswith(".py"):
                    paths.append(os.path.relpath(os.path.join(dirpath, name), root))
    kept = []
    for path in sorted(set(p.replace(os.sep, "/") for p in paths)):
        if any(part in SKIP_DIRS for part in path.split("/")):
            continue
        if not include_tests and is_test_path(path):
            continue
        kept.append(path)
    return kept


def chunk_source(path, source):
    """Yield ``(qualname, start, end, text)`` for each definition plus module-level code."""
    lines = source.split("\n")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return [("<module>", 1, len(lines), source)]

    chunks = []
    covered = [False] * (len(lines) + 2)

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                chunks.append((prefix + child.name, start, end, "\n".join(lines[start - 1 : end])))
                for i in range(start, end + 1):
                    covered[i] = True
            elif isinstance(child, ast.ClassDef):
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                # The class chunk is its header, docstring and attributes: everything
                # up to the first method, so methods are not double counted.
                first_def = next(
                    (
                        n.lineno
                        for n in child.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    ),
                    end + 1,
                )
                head_end = max(start, min(end, first_def - 1))
                chunks.append(
                    (prefix + child.name, start, end, "\n".join(lines[start - 1 : head_end]))
                )
                for i in range(start, head_end + 1):
                    covered[i] = True
                visit(child, prefix + child.name + ".")
                for i in range(start, end + 1):
                    covered[i] = True

    visit(tree, "")
    rest = [lines[i - 1] for i in range(1, len(lines) + 1) if not covered[i]]
    if any(line.strip() for line in rest):
        chunks.append(("<module>", 1, len(lines), "\n".join(rest)))
    return chunks


class Index:
    """BM25 index over code chunks."""

    def __init__(self, root, include_tests=False):
        self.chunks = []  # (path, qualname, start, end)
        self.tfs = []
        self.lengths = []
        self.df = Counter()
        for path in list_python_files(root, include_tests):
            full = os.path.join(root, path)
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    continue
                with open(full, encoding="utf-8", errors="replace") as fh:
                    source = fh.read()
            except OSError:
                continue
            path_tokens = tokenize(path.replace("/", " ").replace(".py", ""))
            for qualname, start, end, text in chunk_source(path, source):
                # Path and symbol name are part of the document, weighted up, so a
                # query naming the module or function matches even if the body does not.
                name_tokens = tokenize(qualname.replace(".", " "))
                tf = Counter(tokenize(text))
                for tok in name_tokens:
                    tf[tok] += 3
                for tok in path_tokens:
                    tf[tok] += 2
                if not tf:
                    continue
                self.chunks.append((path, qualname, start, end))
                self.tfs.append(tf)
                self.lengths.append(sum(tf.values()))
                self.df.update(tf.keys())
        self.n = len(self.chunks)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 1.0

    def idf(self, token):
        df = self.df.get(token, 0)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score_chunks(self, query):
        idents = {i.lower() for i in code_identifiers(query)}
        q_tf = Counter(tokenize(query))
        weights = {}
        for tok, count in q_tf.items():
            if tok not in self.df:
                continue
            # Repeats in the query matter little; code identifiers matter twice as much.
            weight = self.idf(tok) * (1.0 + math.log(count))
            if tok in idents:
                weight *= 2.0
            weights[tok] = weight
        scores = [0.0] * self.n
        for i, tf in enumerate(self.tfs):
            norm = K1 * (1.0 - B + B * self.lengths[i] / self.avg_len)
            total = 0.0
            for tok, weight in weights.items():
                f = tf.get(tok)
                if f:
                    total += weight * f * (K1 + 1.0) / (f + norm)
            scores[i] = total
        return scores

    def rank(self, query, top_files=8, per_file=3):
        """Return ``[(path, file_score, [(qualname, start, end, score), ...]), ...]``."""
        scores = self.score_chunks(query)
        by_file = {}
        for i, score in enumerate(scores):
            if score > 0.0:
                by_file.setdefault(self.chunks[i][0], []).append((score, i))
        ranked = []
        for path, hits in by_file.items():
            hits.sort(reverse=True)
            # Best chunk dominates; further strong chunks in the same file add evidence.
            file_score = hits[0][0] + 0.3 * sum(s for s, _ in hits[1:3])
            symbols = [
                (self.chunks[i][1], self.chunks[i][2], self.chunks[i][3], round(s, 2))
                for s, i in hits[:per_file]
            ]
            ranked.append((path, round(file_score, 2), symbols))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_files]


def format_text(ranked):
    if not ranked:
        return "No matching code found. Try different identifiers from the issue."
    out = ["Most likely places to edit (best first). Read these line ranges with read_file:"]
    for rank, (path, score, symbols) in enumerate(ranked, 1):
        out.append(f"{rank}. {path}  (score {score})")
        for qualname, start, end, _ in symbols:
            out.append(f"     {qualname}  lines {start}-{end}")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("query", nargs="*", help="Issue keywords and identifiers")
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip()
    if not query and not sys.stdin.isatty():
        query = sys.stdin.read()
    if not query.strip():
        print("Usage: locate.py 'keywords and identifiers from the issue'")
        return 0
    ranked = Index(args.root, args.include_tests).rank(query, top_files=args.top)
    if args.json:
        print(json.dumps([{"path": p, "score": s, "symbols": sym} for p, s, sym in ranked]))
    else:
        print(format_text(ranked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
