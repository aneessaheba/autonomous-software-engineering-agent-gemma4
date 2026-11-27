# Local evaluation, shard 0 (43 of 129 public tasks)

First end-to-end runs of the agent with the official harness, on an RTX 5090
(Gemma 4 31B W4A16, vLLM 0.19.1, subprocess sandbox), 25 to 26 November 2026.
Shard 0 is every third public task. Budget: 4.5 minutes and 30 tool calls per
task. Both variants ran on the same 43 tasks.

The figures below were reported from the lab machine; the per-task rows are in
the lab results archive and the full 129-task comparison is still running.

## Resolution rate

| | `agents/baseline` | `agents/locate` |
| :--- | ---: | ---: |
| Resolved | 10 of 43 (23.3%) | 10 of 43 (23.3%) |
| 95% Wilson interval | 13.2 to 37.7% | 13.2 to 37.7% |
| rich / fastapi / requests | 6 of 16, 4 of 23, 0 of 4 | 5 of 16, 5 of 23, 0 of 4 |

Paired on the 43 shared tasks: the skill variant gained one task
(`fastapi_9753`) and lost one (`rich_3480`), net zero. Nine tasks are resolved
by both.

## Behaviour

| | `agents/baseline` | `agents/locate` |
| :--- | ---: | ---: |
| Timeouts at 4.5 minutes | 8 | 2 (plus 1 killed) |
| Empty patches | 5 | 3 |
| Used all 30 tool calls | 12 | 14 |
| Median task time | 143 s | 154 s |
| 90th percentile task time | 278 s | 262 s |
| Total agent time | 1.94 h | 1.96 h |
| Median tokens per task | about 239k | about 273k |

| Test outcome | `agents/baseline` | `agents/locate` |
| :--- | ---: | ---: |
| Tests passed | 10 | 10 |
| Tests ran and failed | 18 | 18 |
| Tests could not be collected | 13 | 14 |
| Tests not run or run killed | 2 | 1 |

## What the runs show

- **The localization skill did not raise the resolution rate on this shard.**
  It cut timeouts from 8 to 2 and empty patches from 5 to 3, but the same
  number of tasks passed. With 43 tasks the interval is wide, so a difference
  of a few points either way would not be visible.
- **41% of skill calls were malformed.** Of 53 `run_skill_script` calls, 22
  were rejected, mostly for omitting `skill_name` and `file_path`. The skill
  was used on 30 of 43 tasks, always as the first action. `agents/locate-v2`
  rewrites the instruction to fix this.
- **About a third of tasks fail before any test runs.** 13 of the 43 tasks have
  hidden tests that import `inline_snapshot`, `dirty_equals` or `orjson`, which
  the sandbox does not provide; a static check finds 27 such tasks among all
  129, all in fastapi. None of the tasks either agent resolved is among them.
  A replay of the reference patches (`python -m gemma_swe.gold_replay`) will
  give the exact count of solvable tasks.
- **Some collection errors are the agent's fault:** it implements the feature
  under a different name or path from the one the hidden tests import, for
  example `scripts/release.py` instead of `scripts/prepare_release.py`.
- **Valid skill calls count against the tool-call budget**; rejected ones do not.
