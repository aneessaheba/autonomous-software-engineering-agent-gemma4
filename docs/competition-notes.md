# Competition notes

Facts about the competition and harness that the code in this repository
depends on. Sources: the Kaggle competition pages and `HARNESS_README.md` from
the competition dataset, read on 2026-11-22, and the harness source in the
organisers' wheelhouse (`swegemma` 0.2.11, `adk-submission` 0.2.13).

## Task and scoring

- The agent receives an issue and a Python repository at `base_commit`, and
  produces a patch. Hidden tests are applied on top; the task is resolved if
  `pytest` exits 0 and every required test passed.
- Score is the resolution rate over about 120 hidden tasks from private
  repositories, split evenly between the public and private leaderboards.
- 129 public tasks are provided for development: fastapi (67), rich (48),
  requests (13), httpx (1).

## Limits

| Limit | Value |
| :--- | :--- |
| Submissions | 1 per day, 2 final selections |
| Final submission deadline | 2 December 2026, 23:59 UTC |
| Whole-run time limit | 12 hours for all tasks, sandbox setup included |
| Scoring hardware | 4 x NVIDIA L4 (96 GB), no internet |
| Context window | 32,768 tokens, prompt + reasoning + output |
| Submission size | under 3 GiB unpacked, adapters included |
| Sandbox | 4 GiB RAM, 2 vCPUs, offline |

Allowed base models (one per submission, shared by every agent):
`gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`,
`gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct`. The two smallest
are eligible for a separate small-model prize.

## Submission format

Declarative only: YAML agent configs, Markdown prompts, optional LoRA adapters
(`.safetensors`) and optional ADK skills. No Python entry point is executed;
skill scripts run inside the task sandbox through `run_skill_script`.

Allowed extensions: `.yaml .yml .md .txt .py .json .safetensors`.

## Harness behaviour that shapes the design

- **Context compaction** starts at 14,336 tokens and keeps the last five events.
  Anything the agent must not forget belongs in the system prompt. The harness
  puts the issue in session state, so `{problem_description}` in the instruction
  pins it. `{hints?}` must use the optional form: most tasks have no hints and
  a required placeholder raises `KeyError`.
- **Any other `{identifier}` in a prompt fails the task** for the same reason.
  The validator rejects them.
- **Test and config edits are discarded** before grading: `test_*.py`,
  `*_test.py`, anything under `tests/`, plus `conftest.py`, `pytest.ini`,
  `pyproject.toml`, `setup.cfg`, `tox.ini`.
- **The working tree is graded without `submit_patch`.** If the budget runs out
  the harness diffs the tree anyway, so an early edit is never wasted.
- **`submit_patch` and `get_status` are free**; the other seven tools each cost
  one tool call. `read_file` returns at most 150 lines or 10,000 characters;
  command output is cut at 5,000 characters.
- **Untracked files in `/workspace` end up in the patch.** Scratch scripts go
  in `/tmp`.

## Findings from reading the harness source

These are not in the published guide and each one would have cost a submission:

1. **Skill names must be kebab-case.** The guide's example is
   `skills/repo_navigation`, but ADK 1.36.1 accepts snake_case only behind a
   feature flag. `compile_submission` rejected `repo_locate`; `repo-locate`
   compiles.
2. **Skill scripts do not start in the repository.** ADK's wrapper changes
   directory to a temp folder before running the script, and the subprocess
   sandbox (the backend the scoring notebook uses) has no `/workspace` mount:
   only shell command strings are rewritten. `locate.py` recovers the root from
   the wrapper's saved working directory.
3. **The harness writes its wrapper into the work tree** as `.adk_exec_*.py`
   while a skill script runs. It contains the whole skill, so an indexer that
   includes hidden files ranks it first for every query.

## Open questions

- Whether `run_skill_script` counts against the tool-call budget. It is not
  wrapped by the harness's budget decorator in the source, which suggests not,
  but this is unverified until a local run.
- How long model start-up and sandbox setup take on the scorer. They count
  toward the 12-hour limit, so the per-task budget has to leave room for them.

## Lessons from submissions

- **An 8-minute per-task budget exceeds the 12-hour limit.** Submission
  58763036 (2026-11-23) was rejected with "submission notebook exceeded the
  allowed runtime" and received no score. Tasks evidently run one after
  another, so 120 tasks at 8 minutes is 16 hours in the worst case. The budget
  is now 4.5 minutes and 30 tool calls, and the validator rejects any config
  whose worst case is above 10 hours.
