# Autonomous Software Engineering Agent on Gemma 4

An agent that resolves real GitHub issues in Python repositories using an open
model that fits on one GPU. Built for Kaggle's
[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)
(Google DeepMind, 2026), where a submission is scored by the share of hidden
issues whose validation tests pass after the agent's patch is applied.

The agent runs offline inside a sandbox with a 32,768-token context, 30 tool
calls and 4.5 minutes per issue. It gets the issue text and a repository; it must
find the fault, edit the code, verify, and submit a patch.

## Results so far

| What was measured | Result | Where |
| :--- | :--- | :--- |
| Issues that name the file to edit | 7.8% of 129 public tasks | [results/dataset_stats.md](results/dataset_stats.md) |
| Median reference fix | 8 added lines, 1 file | [results/dataset_stats.md](results/dataset_stats.md) |
| Fault localization, correct file ranked first | 48.4% with the `repo-locate` skill vs 31.2% for identifier grep | [results/localization.md](results/localization.md) |
| Fault localization, correct file in top 5 | 75.0% vs 50.8% | [results/localization.md](results/localization.md) |
| Function-level: a printed line range overlaps the fix (top 5 files) | 70.3% | [results/localization.md](results/localization.md) |
| Resolution rate, 43-task shard, official harness on an RTX 5090 | 23.3% (10 of 43; 95% CI 13.2 to 37.7%) for both baseline and skill variant | [results/local_eval_shard0.md](results/local_eval_shard0.md) |
| Timeouts on that shard | 2 with the `repo-locate` skill vs 8 without | [results/local_eval_shard0.md](results/local_eval_shard0.md) |
| Full 129-task comparison and Kaggle leaderboard score | not yet measured | runs in progress |

The localization numbers use the full issue text as the query and the directory
prior was chosen on the same 129 tasks, so they are an upper estimate of what
the agent will see when it writes its own query on unseen repositories.

## Approach

Most issues do not say where the bug is, and the reference fix is usually small.
Finding the right function is therefore the main cost, and exploration is what
burns a small model's tool-call and context budget. The design follows from that:

1. **`repo-locate` skill.** A standard-library script shipped inside the
   submission and run in the sandbox. It splits every source file into
   function, method and class chunks with `ast`, ranks them against the issue
   using BM25 over code-aware tokens, and prints the top files with the line
   ranges to read. One tool call replaces several rounds of `grep`.
2. **Pinned issue text.** The issue is repeated at the end of the system prompt
   through ADK session state, so it survives the harness's context compaction.
3. **Budget discipline in the prompt.** Edit by the tenth tool call, verify
   once, submit; the working tree is graded even if the budget runs out.
4. **LoRA fine-tuning** of Gemma 4 on agent trajectories is the next stage.

Two agent variants live under [agents/](agents/) so each idea can be measured
against a control:

| Variant | Description |
| :--- | :--- |
| [agents/baseline](agents/baseline) | Prompt-only single agent, locates code with `git grep` |
| [agents/locate](agents/locate) | Same agent, starts from the `repo-locate` skill |
| [agents/locate-v2](agents/locate-v2) | Fixes found in the first GPU run: unambiguous skill call, a rule for new public names, 40 tool calls. Not yet evaluated |
| [agents/locate-v2-fast](agents/locate-v2-fast) | `locate-v2` with a 1,024-token thinking budget. Not yet evaluated |

## Repository layout

```
agents/            Agent variants; each directory is a complete submission
  baseline/        agent.yaml, prompts/, configs/, eval_config.yaml
  locate/          + skills/repo-locate/ (SKILL.md, scripts/locate.py)
src/gemma_swe/
  tasks.py         Task loading and unified-diff parsing
  submission.py    Validate and package a submission zip
  kaggle_data.py   Download competition data
  worktrees.py     Extract Python sources from repository snapshots
  local_eval.py    Run the official two-phase evaluation on a GPU machine
  gold_replay.py   Replay reference patches to find which tasks are solvable
  analysis/        Dataset statistics, localization benchmark, run reports
scripts/gpu/       Setup and evaluation scripts for a CUDA machine
results/           Tracked numbers (JSON and Markdown)
tests/             Unit tests, including the skill run through ADK's real wrapper
docs/              Competition notes and design decisions
```

Competition data, harness wheels and model weights are not in this repository;
the competition rules forbid redistributing the data.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Kaggle API
credentials for an account that has accepted the competition rules.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[kaggle,dev]"

gemma-swe download core graphs embeddings wheels snapshots   # snapshots are 21.5 GB
gemma-swe stats                    # results/dataset_stats.md
gemma-swe localize                 # results/localization.md
gemma-swe validate agents/locate   # structural checks + official compile
gemma-swe package agents/locate    # dist/submission.zip
pytest
```

To run the official harness checks locally, also install the organisers' wheels:

```bash
gemma-swe download --harness-wheels
uv pip install "google-adk==1.36.1" litellm docker networkx cachetools python-dotenv
uv pip install --no-deps vendor/wheels/*.whl
```

## Evaluating an agent on a GPU

```bash
scripts/gpu/setup.sh                                  # once per machine
scripts/gpu/run_eval.sh agents/baseline baseline-v1   # all 129 public tasks
scripts/gpu/run_eval.sh agents/locate locate-v1
gemma-swe report runs/baseline-v1 runs/locate-v1      # rates, 95% intervals, paired diff
```

`local_eval.py` serves the model with vLLM and drives the organisers' evaluator
with the same compaction and caching settings as the scorer.

## Submission safety

Kaggle allows one submission per day, so every archive is checked before upload:
structure, model and tool whitelists, adapter files, skill manifests and prompt
placeholders, then the organisers' `validate_directory` and a full
`compile_submission`. See [docs/competition-notes.md](docs/competition-notes.md)
for the rules and harness behaviour these checks encode.

## Licence

Apache 2.0, the licence the competition requires of winning solutions.
