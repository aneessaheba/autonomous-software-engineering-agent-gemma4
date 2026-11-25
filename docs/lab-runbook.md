# Lab runbook: first GPU evaluation of the Gemma 4 agent

This file is a brief for the Claude session running on the lab machine. The
person you are helping is the project owner. They are at a university lab with
an NVIDIA RTX 5090 machine and want to be guided **one step at a time**: give
one step, wait for them to confirm or paste output, then give the next. Do not
print the whole plan at them.

## What this project is

A solo entry in Kaggle's "Google - The Gemma 4 Developer Agent Competition"
(https://www.kaggle.com/competitions/gemma-4-developer-agent). The agent gets a
GitHub issue and a Python repository, and must produce a patch; the score is
the share of hidden issues whose tests pass. The final submission deadline is
2 December 2026, 23:59 UTC, with one submission per day.

Repository: https://github.com/aneessaheba/autonomous-software-engineering-agent-gemma4

Read `README.md` and `docs/competition-notes.md` in the repository first. They
describe the layout, the harness and what has been measured.

## Why today matters

Nothing in this project has run the model yet. All existing numbers are offline
analysis done on a laptop. Today's goal is the first real resolution rates:

1. Get the official evaluation harness and Gemma 4 running on the 5090.
2. Run the two agent variants on the public tasks and compare them.
3. Measure how long a task takes, because the one Kaggle submission so far was
   rejected for exceeding the 12-hour run limit.
4. Commit the results and push.

## Rules you must follow

- **Commits:** author must be `aneessaheba <aneessaheba.guddi@sjsu.edu>`. No
  Claude attribution of any kind: no `Co-Authored-By` trailer, no "Generated
  with" line, no mention of Claude in commit messages. Commit in small logical
  units and push to `main`.
- **Never commit competition data, model weights or run directories.** `data/`,
  `vendor/` and `runs/` are gitignored; keep it that way. The competition rules
  forbid redistributing the data. Only curated summaries go in `results/`.
- **Never submit to Kaggle unless the owner tells you to in that moment.** There
  is one slot per day and each one matters.
- **Never print, paste or commit the Kaggle API key.**
- **Report honestly.** If a run fails or a number is low, say so with the
  output. Do not describe anything as working until it has run.

## Facts you need

| Fact | Value |
| :--- | :--- |
| Base model | `gemma-4-31b-it-qat-w4a16-ct`, version 2 on Kaggle Models |
| Model size on disk | 23.3 GB (`model.safetensors`) |
| GPU memory | A 5090 has 32 GB; vLLM is configured to use 90%, about 28.8 GB |
| Competition data | about 22.5 GB, of which 21.5 GB is repository snapshots |
| Free disk needed | at least 70 GB |
| Public tasks | 129 (fastapi 67, rich 48, requests 13, httpx 1) |
| Per-task budget | 4.5 minutes, 30 tool calls (set in each agent's `eval_config.yaml`) |
| Agent variants | `agents/baseline` (prompt only), `agents/locate` (adds the `repo-locate` skill) |
| Scorer's sandbox backend | `subprocess`, which is also the default of the local runner |

**The setup and evaluation scripts have never been run on a GPU.** They were
written on a Mac by transcribing the organisers' starter notebook. Expect at
least one of them to fail, and fix it in the repository when it does (then
commit the fix). The most likely trouble spots are listed under Troubleshooting.

## Steps

### Step 0: check the machine

```bash
nvidia-smi
df -h ~
python3 --version; git --version; tmux -V
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm: a 5090 with about 32 GB is visible and idle, at least 70 GB of disk is
free, and `uv` is installed. If someone else is using the GPU, stop and tell
the owner; the 31B model needs nearly all of it.

### Step 1: Kaggle credentials and model licence

The owner creates a fresh API token on the lab machine instead of copying one:

1. Open https://www.kaggle.com/settings in a browser, log in, and under "API"
   click "Create New Token". A `kaggle.json` file downloads.
2. Move it into place:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Open https://www.kaggle.com/models/google/gemma-4 and accept the licence if
   the page asks for it. Without this the weights download fails with a 403.

Verify (this also proves the account has joined the competition):

```bash
uvx --with six --from kaggle kaggle competitions files gemma-4-developer-agent | head -5
```

Plain `uvx kaggle` fails with a missing `six` module; use the form above.

### Step 2: clone and set the git identity

```bash
git clone https://github.com/aneessaheba/autonomous-software-engineering-agent-gemma4.git
cd autonomous-software-engineering-agent-gemma4
git config user.name "aneessaheba"
git config user.email "aneessaheba.guddi@sjsu.edu"
```

Pushing needs GitHub authentication on this machine (`gh auth login`, or a
personal access token). Check with `git push --dry-run` before doing real work.

### Step 3: one-time setup

```bash
tmux new -s lab
scripts/gpu/setup.sh 2>&1 | tee setup.log
```

This creates `.venv`, installs vLLM and the organisers' pinned wheels, downloads
all competition data (22.5 GB) and the model (23.3 GB). Expect 30 to 90 minutes
depending on the network. It ends by printing the vLLM and torch versions and
the GPU name. If it fails partway, fix the script and re-run it; the data
download skips files that are already complete.

### Step 4: smoke test on three tasks

```bash
scripts/gpu/run_eval.sh agents/baseline smoke --limit 3
```

What should happen: vLLM starts (several minutes the first time), three tasks
run, each prints a line like `[1/3] fastapi_15661 resolved=False calls=12 140.2s`,
and a short report follows. Zero of three resolved is acceptable here; the point
is that the pipeline runs end to end. Read `runs/smoke/run.log` and one file
under `runs/smoke/traces/` to check the agent is making sensible tool calls.

Then run the same three tasks with the skill variant and check the two open
questions listed under "Questions to answer":

```bash
scripts/gpu/run_eval.sh agents/locate smoke-locate --limit 3
```

### Step 5: paired comparison on one shard (about 43 tasks each)

```bash
scripts/gpu/run_eval.sh agents/baseline baseline-s0 --num-shards 3 --shard-index 0
scripts/gpu/run_eval.sh agents/locate   locate-s0   --num-shards 3 --shard-index 0
gemma-swe report runs/baseline-s0 runs/locate-s0
```

Each run takes at most about 3.5 hours (43 tasks at up to 4.5 minutes, plus
verification) and usually much less. This gives a first paired comparison while
the owner is still at the lab. Only one run can use the GPU at a time.

### Step 6: full runs

If time allows, or to leave running overnight inside tmux:

```bash
scripts/gpu/run_eval.sh agents/baseline baseline-v1
scripts/gpu/run_eval.sh agents/locate   locate-v1
gemma-swe report runs/baseline-v1 runs/locate-v1 --out results/local_eval.md
```

A full run is 129 tasks, up to about 10 hours in the worst case. If a run is
interrupted, re-run the same command with `--resume` appended.

### Step 7: record and push

1. Write the report to `results/local_eval.md` with `--out` as above. For shard
   runs, say in the file that it covers one shard of about 43 tasks.
2. Add the measured numbers to the "Results so far" table in `README.md`,
   replacing the "not yet measured" row.
3. Commit `results/` and `README.md` (not `runs/`) and push.

### Step 8: tell the owner what to submit

Using the measured resolution rates and task times, recommend which variant to
submit to Kaggle and with what budget. Package it with
`gemma-swe package agents/<variant> --out dist/submission.zip`. Submit only if
the owner says so.

## Questions to answer from the runs

1. **Does the model call the skill correctly?** In `runs/smoke-locate/traces/`,
   look for a `run_skill_script` call with `skill_name="repo-locate"` and
   `file_path="scripts/locate.py"`, and check that its output lists files. If
   the model calls `load_skill` first or passes wrong arguments, adjust
   `agents/locate/prompts/system.md` and `SKILL.md`.
2. **Does `run_skill_script` count against the 30 tool calls?** Compare the
   `tool_calls` figure with the calls in the trace.
3. **How long does a task really take?** Median and maximum of
   `duration_seconds`, plus the vLLM start-up time and the per-task sandbox
   setup time visible in `run.log`. Kaggle's scorer runs about 120 tasks one
   after another on 4 x L4 GPUs, which are slower than a 5090, inside 12 hours
   total. Use this to judge whether 4.5 minutes per task is safe or can be raised.
4. **Where does the agent fail?** Count empty patches, budget exhaustion and
   wrong-file edits. `gemma-swe report` shows empty patches and medians;
   the per-task rows are in `runs/<name>/task_results.jsonl`.

## Troubleshooting

| Symptom | What to try |
| :--- | :--- |
| `403` downloading the model | Accept the Gemma 4 licence on the Kaggle model page (step 1) |
| vLLM runs out of GPU memory at start-up | The weights are 23.3 GB of a 28.8 GB allowance. Pass `--gpu-memory-utilization 0.95` to `run_eval.sh`; close anything else on the GPU. As a fallback, copy the agent directory, change `model:` to `gemma-4-12b-it-qat-w4a16-ct`, download that model with `MODEL=gemma-4-12b-it-qat-w4a16-ct scripts/gpu/setup.sh`, and say clearly in the results that the 12B model was used |
| vLLM does not know the `gemma4` tool or reasoning parser | The PyPI build differs from the organisers' wheel. Make sure `uv pip install --no-deps --force-reinstall vendor/wheelhouse/*.whl` ran without errors, and compare with the organisers' notebook `ryanholbrook/getting-started-gemma-4-developer-agent` on Kaggle |
| A wheel is rejected as "not a supported wheel" | The wheelhouse targets Python 3.12 on Linux x86_64. Check `python --version` inside `.venv`; skip wheels that do not apply |
| torch does not see the GPU | The 5090 needs a CUDA 12.8 build of torch. Check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |
| `Model weights not found at data/models/...` | The download may have unpacked into a nested folder. The runner expects `data/models/<model-alias>/config.json` |
| Sandbox errors about missing wheels or Python | The subprocess sandbox builds a virtual environment per task from `data/competition/wheels`. Confirm that directory has 124 files and that `python3` is 3.12 or newer |
| A task hangs | Per-command timeout is 120 s and the task budget 4.5 min; if the whole runner hangs, stop it and re-run with `--resume` |

When something fails, read the actual error, fix the cause in the repository,
commit the fix, and carry on. Do not work around a failure by skipping the step.

## What to bring back

- Pushed commits containing `results/local_eval.md`, the updated README table,
  and any script fixes.
- A short note for the owner: resolution rate of each variant with its 95%
  interval, the paired gain or loss, median and maximum task time, answers to
  the four questions above, and your submission recommendation.
