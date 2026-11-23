import zipfile
from pathlib import Path

import pytest

from gemma_swe.submission import package, unknown_template_vars, validate

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = REPO_ROOT / "submission"


def _write_minimal(root: Path, **overrides) -> Path:
    cfg = {
        "name": "a",
        "model": "gemma-4-31b-it-qat-w4a16-ct",
        "instruction": "Fix it. {problem_description}",
        "tools": ["run_command", "submit_patch"],
    }
    cfg.update(overrides)
    lines = []
    for key, value in cfg.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {v}" for v in value)
        else:
            lines.append(f"{key}: {value!r}")
    (root / "agent.yaml").write_text("\n".join(lines) + "\n")
    return root


def test_shipped_submission_is_structurally_valid():
    report = validate(SUBMISSION, use_harness=False)
    assert report.ok, report.errors
    assert report.info["model"] == "gemma-4-31b-it-qat-w4a16-ct"
    assert "submit_patch" in report.info["tools"]


def test_shipped_submission_compiles_under_official_harness():
    pytest.importorskip("adk_submission")
    pytest.importorskip("swegemma")
    report = validate(SUBMISSION, use_harness=True)
    assert report.ok, report.errors
    assert report.info["harness_compile"].startswith("ok")


def test_shipped_prompt_pins_the_issue_text():
    prompt = (SUBMISSION / "prompts" / "system.md").read_text()
    assert "{problem_description}" in prompt
    assert unknown_template_vars(prompt) == []


@pytest.mark.parametrize(
    ("text", "n_bad"),
    [
        ("plain text", 0),
        ("{problem_description} and {hints?}", 0),
        ("{hints}", 1),  # required form fails on tasks without hints
        ("{repo_name}", 1),
        ('json like {"a": 1} is left alone', 0),
    ],
)
def test_unknown_template_vars(text, n_bad):
    assert len(unknown_template_vars(text)) == n_bad


def test_rejects_unknown_tool(tmp_path):
    _write_minimal(tmp_path, tools=["run_command", "browse_web"])
    report = validate(tmp_path, use_harness=False)
    assert any("unknown tool 'browse_web'" in e for e in report.errors)


def test_rejects_disallowed_model(tmp_path):
    _write_minimal(tmp_path, model="gemma-4-27b-it")
    report = validate(tmp_path, use_harness=False)
    assert any("is not one of" in e for e in report.errors)


def test_rejects_disallowed_extension_and_missing_root(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"x")
    report = validate(tmp_path, use_harness=False)
    assert any("Disallowed file extension" in e for e in report.errors)
    assert any("root config" in e for e in report.errors)


def test_rejects_missing_adapter_weights(tmp_path):
    _write_minimal(tmp_path, adapter="main_lora")
    report = validate(tmp_path, use_harness=False)
    assert any("missing adapter_model.safetensors" in e for e in report.errors)


def test_small_model_is_flagged_bonus_eligible(tmp_path):
    _write_minimal(tmp_path, model="gemma-4-e4b-it-qat-w4a16-ct")
    assert validate(tmp_path, use_harness=False).info["small_model_bonus_eligible"]


def test_package_is_deterministic_with_root_config_at_archive_root(tmp_path):
    first = package(SUBMISSION, tmp_path / "a.zip")
    second = package(SUBMISSION, tmp_path / "b.zip")
    assert first["content_sha256"] == second["content_sha256"]
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "a.zip") as zf:
        assert "agent.yaml" in zf.namelist()
        assert zf.testzip() is None
