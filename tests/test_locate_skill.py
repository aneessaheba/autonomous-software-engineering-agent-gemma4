"""Tests for the repo-locate skill, including the exact path the harness runs it through."""

import subprocess
import sys
from pathlib import Path

import pytest

from gemma_swe.analysis.localization import LOCATE_SCRIPT, load_locate

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / LOCATE_SCRIPT.parents[1]

FILES = {
    "pkg/__init__.py": "",
    "pkg/cookies.py": (
        "class CookieJar:\n"
        '    """Stores cookies."""\n\n'
        "    def set_cookie(self, name, value):\n"
        "        self.jar[name] = value\n\n"
        "    def clear_expired_cookies(self):\n"
        "        raise CookieConflictError('duplicate cookie')\n"
    ),
    "pkg/sessions.py": "def open_session(url):\n    return url\n",
    "tests/test_cookies.py": "def test_clear_expired_cookies():\n    assert True\n",
}


@pytest.fixture
def repo(tmp_path):
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.fixture(scope="module")
def locate():
    return load_locate(REPO_ROOT / LOCATE_SCRIPT)


def test_tokenize_splits_snake_and_camel_case(locate):
    tokens = locate.tokenize("clear_expired_cookies CookieConflictError")
    assert {
        "clear_expired_cookies",
        "expired",
        "cookies",
        "cookieconflicterror",
        "conflict",
    } <= set(tokens)
    assert "the" not in locate.tokenize("the self return")


def test_code_identifiers_picks_marked_code(locate):
    idents = locate.code_identifiers("Calling `jar.set_cookie` raises CookieConflictError today")
    assert {"set_cookie", "CookieConflictError"} <= idents
    assert "Calling" not in idents


def test_chunks_cover_methods_with_line_ranges(locate):
    chunks = {q: (s, e) for q, s, e, _ in locate.chunk_source("x.py", FILES["pkg/cookies.py"])}
    assert chunks["CookieJar.set_cookie"] == (4, 5)
    assert chunks["CookieJar.clear_expired_cookies"] == (7, 8)
    assert chunks["CookieJar"] == (1, 8)


def test_ranks_the_right_file_and_function_first(locate, repo):
    ranked = locate.Index(str(repo)).rank("clear_expired_cookies raises CookieConflictError")
    path, _, symbols = ranked[0]
    assert path == "pkg/cookies.py"
    assert symbols[0][0] == "CookieJar.clear_expired_cookies"


def test_tests_are_excluded_unless_requested(locate, repo):
    default = {p for p, _, _ in locate.Index(str(repo)).rank("clear expired cookies")}
    with_tests = {p for p, _, _ in locate.Index(str(repo), True).rank("clear expired cookies")}
    assert "tests/test_cookies.py" not in default
    assert "tests/test_cookies.py" in with_tests


def test_runs_through_the_real_adk_skill_wrapper(repo):
    """Reproduce the harness: ADK wraps the script, the sandbox runs it from the repo root."""
    pytest.importorskip("google.adk")
    from google.adk.skills import load_skill_from_dir
    from google.adk.tools.skill_toolset import _SkillScriptCodeExecutor

    skill = load_skill_from_dir(SKILL_DIR)
    wrapper = _SkillScriptCodeExecutor(base_executor=None, script_timeout=60)._build_wrapper_code(
        skill, "scripts/locate.py", ["clear_expired_cookies CookieConflictError"]
    )
    # AdkSandboxCodeExecutor writes the wrapper into the work dir as a hidden .py file.
    (repo / ".adk_exec_deadbeef.py").write_text(wrapper)
    result = subprocess.run(
        [sys.executable, ".adk_exec_deadbeef.py"], cwd=repo, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[1].startswith("1. pkg/cookies.py"), result.stdout
    assert ".adk_exec" not in result.stdout
