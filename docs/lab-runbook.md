# Lab runbook: first GPU evaluation of the Gemma 4 agent

This file is a brief for the Claude assistant helping at the university GPU
lab. The person you are helping is the project owner. Guide them **one step at
a time**: give one step, wait for them to confirm or paste the output, then give
the next. Do not print the whole plan at them.

You are most likely running in a browser on the lab machine and cannot run
commands yourself. The owner runs each command and pastes back what it printed.
Ask for the output whenever a step can fail.

## The lab environment

From the lab's own user guide (`4_Lab_Docker_UserGuide.pdf`, in the same Drive
folder as this file; read it too):

- The workstation runs **Windows** with **Docker Desktop**. All work happens
  inside a Docker container started from an image the lab provides
  (`C:\DockerImages`, for example `pytorch:latest`).
- **Do not install software on the Windows machine** and do not change its
  settings. Installing Python packages *inside the container* is the intended
  way to work.
- **Files may only be created in one folder on the Desktop**, named after the
  owner. That folder is mounted inside the container at `/app`.
- **No USB drives.** Everything arrives by download on the workstation.
- At the end, **delete the Desktop folder** and lock the machine. Results must
  be collected before 12:30 pm on the last day of the reservation.

## What this project is

A solo entry in Kaggle's "Google - The Gemma 4 Developer Agent Competition"
(https://www.kaggle.com/competitions/gemma-4-developer-agent). The agent gets a
GitHub issue and a Python repository and must produce a patch; the score is the
share of hidden issues whose tests pass. The final submission deadline is
2 December 2026, 23:59 UTC, with one submission per day.

Repository: https://github.com/aneessaheba/autonomous-software-engineering-agent-gemma4

`README.md` and `docs/competition-notes.md` in the repository describe the
layout, the harness and what has been measured.

## Why today matters

Nothing in this project has run the model yet. Every existing number is offline
analysis done on a laptop. Today's goals, in order:

1. Get the official evaluation harness and Gemma 4 running on the lab GPU.
2. Run the two agent variants on the public tasks and compare them.
3. Measure how long a task takes. The one Kaggle submission so far was rejected
   for exceeding the 12-hour run limit.
4. Get the results off the machine before the folder is deleted.

## Rules you must follow

- **Follow the lab guide.** If anything here conflicts with it, the lab guide wins.
- **No GitHub login and no Kaggle submission from the lab machine.** It is a
  shared computer. Results leave as an archive uploaded to Google Drive through
  the browser; the owner commits them later from their own laptop.
- **The Kaggle API key** is used only inside the container and the Desktop
  folder. Never paste it into a chat. It is expired at the end (step 10).
- **Report honestly.** If a run fails or a number is low, say so with the
  output. Do not call anything working until it has run.
- **If a script fails, fix the cause.** Edit the file in the project folder, note
  what you changed and why, and carry on. Changed files are included in the
  results archive so the fix is not lost.

## Facts you need

| Fact | Value |
| :--- | :--- |
| Base model | `gemma-4-31b-it-qat-w4a16-ct`, version 2 on Kaggle Models |
| Model size on disk | 23.3 GB |
| GPU memory | An RTX 5090 has 32 GB; vLLM is set to use 90%, about 28.8 GB |
| Competition data | about 22.5 GB, of which 21.5 GB is repository snapshots |
| Free disk needed on C: | at least 80 GB |
| Public tasks | 129 (fastapi 67, rich 48, requests 13, httpx 1) |
| Per-task budget | 4.5 minutes, 30 tool calls (each agent's `eval_config.yaml`) |
| Agent variants | `agents/baseline` (prompt only), `agents/locate` (adds the `repo-locate` skill) |
| Sandbox backend | `subprocess`: no Docker-in-Docker is needed |

**The setup and evaluation scripts have never been run on a GPU, and never in
this lab's container.** They were written on a Mac from the organisers' starter
notebook. Expect at least one failure; the troubleshooting table lists the
likely ones.

## Steps

### Step 1: prepare the workstation (Windows)

Follow steps 1 to 3 of the lab guide:

1. Log in, start Docker Desktop, and load the `pytorch` image by double-clicking
   it in `C:\DockerImages`.
2. Create a folder on the Desktop named after the owner. This is the working
   directory. Check in File Explorer that drive C: has at least 80 GB free.

### Step 2: download the project and a Kaggle key into that folder

All in the workstation's browser:

1. Open the Drive folder "Gemma 4 Agent - Lab Pack", download the
   `repo-snapshot-*.zip` file, and extract it into the Desktop folder. The
   result should be `Desktop\<name>\autonomous-software-engineering-agent-gemma4\`.
   Use the zip, not a `git clone` on Windows, so the shell scripts keep their
   Linux line endings.
2. Open https://www.kaggle.com/settings, and under "API" click "Create New
   Token". Move the downloaded `kaggle.json` into the Desktop folder.
3. Open https://www.kaggle.com/models/google/gemma-4 and accept the licence if
   the page asks. Without this the model download fails with a 403.

### Step 3: start the container

In **Command Prompt** (not PowerShell), using the folder path from File
Explorer and the image name and tag shown in Docker Desktop:

```
docker run --gpus all --ipc=host -p 8888:8888 -v "C:\Users\<user>\Desktop\<name>":/app pytorch:latest
```

This is the lab guide's command with one addition, `--ipc=host`, which gives
the model server enough shared memory. Leave this window open for the whole
session; closing it stops the container. Open the Jupyter link it prints, then
choose New > Terminal. All later commands go in that Jupyter terminal.

### Step 4: check the container

```bash
nvidia-smi
python3 --version
df -h /app /
free -g
ls /app
```

Confirm: an RTX 5090 with about 32 GB is listed, `/app` shows the project
folder and `kaggle.json`, and there is at least 80 GB free on `/app`. Note the
Python version and total memory; the owner should paste this output.

### Step 5: install the Kaggle key inside the container

```bash
mkdir -p ~/.kaggle
cp /app/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 6: one-time setup

```bash
cd /app/autonomous-software-engineering-agent-gemma4
export VENV_DIR=/opt/venv
nohup bash scripts/gpu/setup.sh > setup.log 2>&1 &
tail -f setup.log
```

`VENV_DIR=/opt/venv` keeps the Python environment on the container's own disk,
because `/app` is a Windows folder where symlinks do not work. `nohup` keeps the
job alive if the browser tab closes; `Ctrl+C` stops `tail` only, not the job.

The script installs vLLM and the organisers' pinned packages, then downloads the
competition data (22.5 GB) and the model (23.3 GB). Expect 30 to 90 minutes. It
finishes by printing the vLLM and torch versions and the GPU name. If it fails,
fix the script and run the same command again; completed downloads are skipped.

In every new terminal, run the first two lines again (`cd ...` and `export
VENV_DIR=/opt/venv`) before anything else.

### Step 7: smoke test on three tasks

```bash
nohup bash scripts/gpu/run_eval.sh agents/baseline smoke --limit 3 > smoke.log 2>&1 &
tail -f smoke.log
```

What should happen: vLLM starts (several minutes the first time), three tasks
run, each prints a line like `[1/3] fastapi_15661 resolved=False calls=12 140.2s`,
and a short report follows. Zero of three resolved is acceptable; the point is
that the pipeline runs end to end.

Then the skill variant:

```bash
nohup bash scripts/gpu/run_eval.sh agents/locate smoke-locate --limit 3 > smoke-locate.log 2>&1 &
tail -f smoke-locate.log
```

Have the owner paste the last 40 lines of each log, and answer questions 1 and
2 under "Questions to answer" before going further.

### Step 8: paired comparison on one shard (43 tasks each)

One command runs both variants back to back, unattended:

```bash
nohup bash -c "bash scripts/gpu/run_eval.sh agents/baseline baseline-s0 --num-shards 3 --shard-index 0; bash scripts/gpu/run_eval.sh agents/locate locate-s0 --num-shards 3 --shard-index 0" > shard.log 2>&1 &
tail -f shard.log
```

Each variant takes at most about 3.5 hours and usually much less. Only one run
can use the GPU at a time. When both finish:

```bash
source /opt/venv/bin/activate
gemma-swe report runs/baseline-s0 runs/locate-s0 --out results/local_eval_shard0.md
```

### Step 9: full runs, if the reservation allows

Ask the owner how long the reservation lasts. A full run is 129 tasks and up to
about 10 hours per variant, so both need roughly a day. To leave them running,
the Command Prompt window and Docker Desktop must stay open and the user must
stay logged in; lock the screen with Windows+L instead of signing out.

```bash
nohup bash -c "bash scripts/gpu/run_eval.sh agents/baseline baseline-v1; bash scripts/gpu/run_eval.sh agents/locate locate-v1" > full.log 2>&1 &
```

If a run is interrupted, run the same `run_eval.sh` line again with `--resume`
appended. Afterwards:

```bash
gemma-swe report runs/baseline-v1 runs/locate-v1 --out results/local_eval.md
```

### Step 10: get the results out, then clean up

1. Archive the results and any files that were changed:
   ```bash
   cd /app/autonomous-software-engineering-agent-gemma4
   tar czf /app/lab-results.tgz --exclude='runs/*/agent_config' runs results scripts src agents *.log
   ls -lh /app/lab-results.tgz
   ```
2. In the workstation's browser, upload `lab-results.tgz` from the Desktop
   folder to the Drive folder "Gemma 4 Agent - Lab Pack". Confirm it appears
   there before deleting anything.
3. On https://www.kaggle.com/settings, expire the API token created in step 2.
4. Follow step 7 of the lab guide: close the windows, stop the container,
   **delete the Desktop folder**, and lock the machine.

## Questions to answer from the runs

1. **Does the model call the skill correctly?** In `runs/smoke-locate/traces/`,
   look for a `run_skill_script` call with `skill_name="repo-locate"` and
   `file_path="scripts/locate.py"` whose output lists files:
   `grep -o 'run_skill_script[^}]*' runs/smoke-locate/traces/*.json | head`.
   If the model calls `load_skill` first or passes wrong arguments, adjust
   `agents/locate/prompts/system.md` and the skill's `SKILL.md`.
2. **Does `run_skill_script` count against the 30 tool calls?** Compare the
   `tool_calls` figure in `task_results.jsonl` with the calls in the trace.
3. **How long does a task really take?** Median and maximum
   `duration_seconds`, the vLLM start-up time, and the per-task sandbox setup
   time visible in `run.log`. Kaggle runs about 120 tasks one after another on
   4 x L4 GPUs, which are slower than a 5090, within 12 hours in total. Judge
   whether 4.5 minutes per task is safe or could be raised.
4. **Where does the agent fail?** Count empty patches, exhausted budgets and
   edits to the wrong file. `gemma-swe report` gives empty patches and medians;
   per-task rows are in `runs/<name>/task_results.jsonl`.

## Troubleshooting

| Symptom | What to try |
| :--- | :--- |
| `bash\r: command not found` or `$'\r'` errors | The scripts got Windows line endings. Run `sed -i 's/\r$//' scripts/gpu/*.sh` |
| `403` downloading the model | Accept the Gemma 4 licence on the Kaggle model page (step 2) |
| Container cannot see the GPU | Restart Docker Desktop, or the machine as the lab guide suggests, then repeat step 3 |
| vLLM runs out of GPU memory at start-up | The weights are 23.3 GB of a 28.8 GB allowance. Append `--gpu-memory-utilization 0.95` to the `run_eval.sh` command. As a fallback use the 12B model: copy an agent folder, change `model:` to `gemma-4-12b-it-qat-w4a16-ct`, download it with `MODEL=gemma-4-12b-it-qat-w4a16-ct bash scripts/gpu/setup.sh`, and state clearly in the results that 12B was used |
| vLLM is killed while loading, or the container exits | Docker Desktop's memory limit may be below what the 23 GB model needs. Report `free -g` to the owner; the limit is a machine setting that only lab staff may change, so fall back to the 12B model |
| vLLM does not know the `gemma4` tool or reasoning parser | The PyPI build differs from the organisers' wheel. Check that the `--force-reinstall vendor/wheelhouse/*.whl` line in `setup.log` ran without errors |
| A wheel is "not a supported wheel on this platform" | The wheelhouse targets Python 3.12 on Linux x86_64. Check `/opt/venv/bin/python --version`; delete wheels that do not apply and re-run |
| torch does not see the GPU inside the environment | A 5090 needs a CUDA 12.8 build of torch. Check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |
| `Model weights not found at data/models/...` | The download may have unpacked into a nested folder. The runner expects `data/models/<model-alias>/config.json` |
| Downloads or installs into `/app` are very slow | `/app` is a Windows folder shared into the container. Large files are fine; if the sandbox is slow, confirm its temp files are under `/tmp`, not `/app` |
| `No space left on device` | Check `df -h /app /`. The environment in `/opt/venv` needs about 12 GB on the container disk; data and model need about 46 GB on `/app` |
| The container was stopped and `/opt/venv` is gone | Start it again (step 3), redo steps 5 and 6. Downloads in `/app` are kept, so only the install repeats |

## What to bring back

- `lab-results.tgz` uploaded to the Drive folder.
- A short note for the owner: resolution rate of each variant with its 95%
  interval, the paired gain or loss, median and maximum task time, answers to
  the four questions, every file you changed and why, and which variant and
  budget you recommend submitting to Kaggle.
