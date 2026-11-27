#!/usr/bin/env bash
# Evaluate one agent variant on the public tasks with the official harness.
#
#   scripts/gpu/run_eval.sh agents/locate locate-v1              # all 129 tasks
#   scripts/gpu/run_eval.sh agents/baseline smoke --limit 3      # quick check
#   scripts/gpu/run_eval.sh agents/locate locate-v1 --resume     # continue a run
#
# Results land in runs/<run-name>/; summarise with `gemma-swe report runs/<run-name>`.
set -euo pipefail

AGENT_DIR="${1:?usage: run_eval.sh <agent-dir> <run-name> [local_eval options]}"
RUN_NAME="${2:?usage: run_eval.sh <agent-dir> <run-name> [local_eval options]}"
shift 2

# Find the environment: VENV_DIR if set, else ./.venv, else the lab container's /opt/venv.
if [ -z "${VENV_DIR:-}" ]; then
  if [ -f .venv/bin/activate ]; then VENV_DIR=.venv; else VENV_DIR=/opt/venv; fi
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
gemma-swe validate "$AGENT_DIR"

RESULTS="runs/$RUN_NAME"
mkdir -p "$RESULTS"
git rev-parse HEAD > "$RESULTS/git_commit.txt" 2>/dev/null || true
cp -R "$AGENT_DIR" "$RESULTS/agent_config"

python -m gemma_swe.local_eval "$AGENT_DIR" --results-dir "$RESULTS" "$@" 2>&1 | tee -a "$RESULTS/run.log"
gemma-swe report "$RESULTS"
