"""Validate and package the agent submission.

Kaggle allows one submission per day, so a zip that fails to compile wastes a
whole day. ``validate`` therefore runs three layers of checks before anything
is uploaded:

1. Structural checks implemented here, which need no harness install.
2. The organisers' own ``adk_submission.validate_directory`` when available.
3. A full ``compile_submission`` into an ADK agent tree against stub tool and
   model registries, which is what the scorer does before running any task.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ALLOWED_MODELS = frozenset(
    {
        "gemma-4-31b-it-qat-w4a16-ct",
        "gemma-4-12b-it-qat-w4a16-ct",
        "gemma-4-e4b-it-qat-w4a16-ct",
        "gemma-4-e2b-it-qat-w4a16-ct",
    }
)
SMALL_MODEL_BONUS = frozenset({"gemma-4-e4b-it-qat-w4a16-ct", "gemma-4-e2b-it-qat-w4a16-ct"})
HARNESS_TOOLS = frozenset(
    {
        "run_command",
        "submit_patch",
        "get_status",
        "read_file",
        "edit_file",
        "write_file",
        "get_code_neighbors",
        "search_similar_code",
        "get_code_subgraph",
    }
)
ALLOWED_EXTENSIONS = frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"})
ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
MAX_TOTAL_BYTES = 3 * 1024**3
MAX_CONTEXT_TOKENS = 32_768
# Variables the harness puts in ADK session state; any other {identifier} in an
# instruction raises KeyError at run time and fails the task.
SESSION_STATE_VARS = frozenset({"problem_description", "hints"})
_TEMPLATE = re.compile(r"{+[^{}]*}+")
_KEBAB = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# Fixed timestamp so the same config always produces a byte-identical archive.
_ZIP_EPOCH = (2026, 1, 1, 0, 0, 0)


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader that resolves the harness's ``!include`` tag relative to the file."""


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_yaml(path: Path, root: Path, depth: int = 0):
    if depth > 10:
        raise ValueError(f"!include depth exceeds 10 at {path}")

    def include(loader: yaml.SafeLoader, node: yaml.Node):
        rel = loader.construct_scalar(node)
        if Path(rel).is_absolute():
            raise ValueError(f"!include path must be relative: {rel}")
        # ``..`` is fine (sub_agents/x.yaml includes ../prompts/y.md) as long as
        # the resolved target stays inside the submission root.
        target = (path.parent / rel).resolve()
        if root.resolve() not in target.parents:
            raise ValueError(f"!include path escapes the submission root: {rel}")
        if not target.is_file():
            raise FileNotFoundError(f"!include target not found: {rel} (from {path.name})")
        if target.suffix in {".yaml", ".yml"}:
            return _load_yaml(target, root, depth + 1)
        return target.read_text(encoding="utf-8")

    loader_cls = type("Loader", (_IncludeLoader,), {})
    loader_cls.add_constructor("!include", include)
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=loader_cls)  # noqa: S506 - SafeLoader subclass


def find_root_config(submission_dir: Path) -> Path:
    found = [submission_dir / n for n in ROOT_CONFIG_NAMES if (submission_dir / n).is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one root config, found {[p.name for p in found]}")
    return found[0]


def _walk_agents(cfg: dict, cfg_path: Path, root: Path, seen: set[Path]):
    """Yield ``(config_dict, path)`` for the agent and everything it references."""
    yield cfg, cfg_path
    refs: list[str] = []
    for sub in cfg.get("sub_agents") or []:
        if isinstance(sub, dict) and "config_path" in sub:
            refs.append(sub["config_path"])
        elif isinstance(sub, dict):
            yield from _walk_agents(sub, cfg_path, root, seen)
    for tool in cfg.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("agent_tool"), dict):
            refs.append(tool["agent_tool"]["config_path"])
    for rel in refs:
        target = (cfg_path.parent / rel).resolve()
        if target in seen:
            continue
        seen.add(target)
        yield from _walk_agents(_load_yaml(target, root), target, root, seen)


def unknown_template_vars(instruction: str) -> list[str]:
    """Return ``{identifier}`` placeholders the harness will not be able to fill."""
    bad = []
    for match in _TEMPLATE.finditer(instruction):
        name = match.group().lstrip("{").rstrip("}").strip()
        optional = name.endswith("?")
        name = name.removesuffix("?")
        if not name.isidentifier():
            continue  # ADK leaves non-identifier braces untouched
        if name not in SESSION_STATE_VARS:
            bad.append(match.group())
        elif name == "hints" and not optional:
            bad.append(match.group() + " (hints is absent for most tasks; use {hints?})")
    return bad


def _structural_checks(submission_dir: Path, report: ValidationReport) -> None:
    total = 0
    for path in sorted(submission_dir.rglob("*")):
        rel = path.relative_to(submission_dir)
        if path.is_symlink():
            report.errors.append(f"Symlinks are not permitted: {rel}")
            continue
        if path.is_file():
            total += path.stat().st_size
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                report.errors.append(f"Disallowed file extension: {rel}")
    report.info["total_bytes"] = total
    if total >= MAX_TOTAL_BYTES:
        report.errors.append(f"Submission is {total} bytes; the limit is {MAX_TOTAL_BYTES}")

    try:
        root_cfg = find_root_config(submission_dir)
        cfg = _load_yaml(root_cfg, submission_dir)
        agents = list(_walk_agents(cfg, root_cfg, submission_dir, {root_cfg.resolve()}))
    except Exception as exc:  # report, do not crash: the caller wants the full list
        report.errors.append(f"Config failed to load: {exc}")
        return

    models: set[str] = set()
    adapters: set[str] = set()
    tools: set[str] = set()
    for agent, path in agents:
        name = agent.get("name", path.name)
        if agent.get("agent_class", "LlmAgent") != "LlmAgent":
            continue
        if agent.get("model"):
            models.add(str(agent["model"]).split("/")[-1])
        if agent.get("adapter"):
            adapters.add(agent["adapter"])
        for tool in agent.get("tools") or []:
            if isinstance(tool, str):
                tools.add(tool)
                if tool not in HARNESS_TOOLS:
                    report.errors.append(f"{name}: unknown tool {tool!r}")
        for bad in unknown_template_vars(str(agent.get("instruction") or "")):
            report.errors.append(f"{name}: instruction placeholder {bad} is not in session state")
        gen = agent.get("generate_content_config") or {}
        out_tokens = gen.get("max_output_tokens", 16_384)
        budget = (gen.get("thinking_config") or {}).get("thinking_budget", 4_096)
        if not 1 <= out_tokens <= MAX_CONTEXT_TOKENS:
            report.errors.append(f"{name}: max_output_tokens {out_tokens} outside 1..32768")
        if not 0 <= budget <= MAX_CONTEXT_TOKENS:
            report.errors.append(f"{name}: thinking_budget {budget} outside 0..32768")
        if budget >= out_tokens:
            report.warnings.append(
                f"{name}: thinking_budget {budget} >= max_output_tokens {out_tokens}; "
                "a long thought can truncate the tool call"
            )

    if len(models) != 1:
        report.errors.append(f"Exactly one base model must be declared, found {sorted(models)}")
    elif not models <= ALLOWED_MODELS:
        report.errors.append(f"Model {sorted(models)} is not one of {sorted(ALLOWED_MODELS)}")
    for adapter in sorted(adapters):
        adir = submission_dir / "adapters" / adapter
        for required in ("adapter_config.json", "adapter_model.safetensors"):
            if not (adir / required).is_file():
                report.errors.append(f"Adapter {adapter!r} is missing {required}")
    skill_dirs = sorted({s for a, _ in agents for s in (a.get("skills") or [])})
    for rel in skill_dirs:
        manifest = submission_dir / rel / "SKILL.md"
        if not manifest.is_file():
            report.errors.append(f"Skill {rel!r} has no SKILL.md")
            continue
        text = manifest.read_text(encoding="utf-8")
        front = yaml.safe_load(text.split("---")[1]) if text.startswith("---") else None
        if not isinstance(front, dict) or front.get("name") != Path(rel).name:
            report.errors.append(f"Skill {rel!r}: frontmatter name must equal the directory name")
        elif not _KEBAB.fullmatch(front["name"]):
            # ADK only accepts snake_case skill names behind a feature flag.
            report.errors.append(f"Skill {rel!r}: name must be lowercase kebab-case")
        elif not str(front.get("description") or "").strip():
            report.errors.append(f"Skill {rel!r}: frontmatter description is empty")
    if "submit_patch" not in tools:
        report.warnings.append("No agent declares submit_patch; only the fallback diff is graded")

    report.info.update(
        model=next(iter(models), None),
        small_model_bonus_eligible=bool(models) and models <= SMALL_MODEL_BONUS,
        n_agents=len(agents),
        tools=sorted(tools),
        adapters=sorted(adapters),
        skills=skill_dirs,
        instruction_chars=sum(len(str(a.get("instruction") or "")) for a, _ in agents),
    )

    eval_cfg_path = submission_dir / "eval_config.yaml"
    if eval_cfg_path.is_file():
        section = (yaml.safe_load(eval_cfg_path.read_text()) or {}).get("evaluation", {})
        report.info["budget"] = section
        minutes = float(section.get("max_time_minutes", 60))
        # ~120 hidden tasks must finish in 12 h including sandbox setup.
        worst_case_hours = 120 * minutes / 60
        report.info["worst_case_hours_120_tasks"] = round(worst_case_hours, 1)
        if worst_case_hours > 24:
            report.warnings.append(
                f"Worst case {worst_case_hours:.0f} h for 120 tasks is far beyond the 12 h limit"
            )
    else:
        report.warnings.append("No eval_config.yaml: harness defaults (60 min, 100 calls) apply")


def _harness_checks(submission_dir: Path, report: ValidationReport) -> None:
    try:
        from adk_submission import ModelRegistry, compile_submission, validate_directory
        from swegemma.config import build_submission_limits
    except ImportError:
        report.warnings.append("Harness not installed: skipped official validation and compile")
        report.info["harness_compile"] = "skipped"
        return

    limits, constraints = build_submission_limits()
    try:
        validate_directory(submission_dir, limits=limits)
    except Exception as exc:
        report.errors.append(f"validate_directory: {type(exc).__name__}: {exc}")
        return

    def _stub(name: str):
        def tool(**kwargs) -> str:
            return "{}"

        tool.__name__ = name
        tool.__doc__ = f"Stub for {name}."
        return tool

    models = ModelRegistry()
    for alias in ALLOWED_MODELS:
        models.register(alias, f"openai/{alias}")
    try:
        agent = compile_submission(
            submission_dir=submission_dir,
            tool_registry={name: _stub(name) for name in HARNESS_TOOLS},
            model_registry=models,
            limits=limits,
            generation_constraints=constraints,
            adapter_resolver_fn=lambda base, info: base,
        )
    except Exception as exc:
        report.errors.append(f"compile_submission: {type(exc).__name__}: {exc}")
        return
    report.info["harness_compile"] = f"ok ({type(agent).__name__} {agent.name!r})"


_STAGE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".ipynb_checkpoints")


@contextmanager
def staged(submission_dir: str | Path):
    """Yield a clean copy of the submission, as it will appear inside the zip.

    Local tooling leaves ``__pycache__`` and ``.DS_Store`` behind; the scorer
    rejects those extensions, so every check and the archive use this copy.
    """
    with tempfile.TemporaryDirectory(prefix="submission_") as tmp:
        dest = Path(tmp) / "submission"
        shutil.copytree(submission_dir, dest, ignore=_STAGE_IGNORE, symlinks=True)
        yield dest


def validate(submission_dir: str | Path, *, use_harness: bool = True) -> ValidationReport:
    report = ValidationReport()
    if not Path(submission_dir).is_dir():
        report.errors.append(f"Not a directory: {submission_dir}")
        return report
    with staged(submission_dir) as clean:
        _structural_checks(clean, report)
        if use_harness and report.ok:
            _harness_checks(clean, report)
    return report


def package(submission_dir: str | Path, out_path: str | Path) -> dict:
    """Write a deterministic ``submission.zip`` with the root config at archive root."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with (
        staged(submission_dir) as clean,
        zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf,
    ):
        files = sorted(p for p in clean.rglob("*") if p.is_file())
        names = [p.relative_to(clean).as_posix() for p in files]
        for path, arcname in zip(files, names, strict=True):
            data = path.read_bytes()
            info = zipfile.ZipInfo(arcname, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
            digest.update(arcname.encode() + b"\0" + data + b"\0")
    return {
        "path": str(out_path),
        "files": names,
        "zip_bytes": out_path.stat().st_size,
        "content_sha256": digest.hexdigest(),
    }
