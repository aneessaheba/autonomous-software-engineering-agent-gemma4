#!/usr/bin/env bash
# One-time setup of a Linux CUDA machine (tested target: a single RTX 5090, 32 GB)
# for local evaluation and LoRA training. Run from the repository root.
#
# Requires: NVIDIA driver with CUDA 12.8 support, uv, and Kaggle API credentials
# in ~/.kaggle (the account must have accepted the competition rules).
set -euo pipefail

MODEL="${MODEL:-gemma-4-31b-it-qat-w4a16-ct}"
MODEL_VERSION="${MODEL_VERSION:-2}"
WHEELHOUSE="vendor/wheelhouse"

command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The harness requires Python >= 3.12; the scorer's wheels are built for cp312.
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[kaggle,dev,plots]"

# 1. Organisers' wheelhouse: vLLM, transformers and the three harness packages,
#    pinned to exactly what the scorer runs.
mkdir -p "$WHEELHOUSE"
kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -p "$WHEELHOUSE" --unzip
# Kaggle strips the "+" from local version tags; restore it so pip accepts the wheels.
for w in "$WHEELHOUSE"/*cu128*.whl; do
  [ -e "$w" ] || continue
  case "$w" in *+cu128*) ;; *) mv "$w" "${w/cu128/+cu128}" ;; esac
done

# 2. Resolve dependencies from PyPI first, then overwrite with the pinned wheels.
uv pip install "vllm==0.19.1" "google-adk==1.36.1" litellm docker networkx cachetools \
  python-dotenv pyarrow peft accelerate
uv pip install --no-deps --force-reinstall "$WHEELHOUSE"/*.whl

# 3. Competition data: everything, including the 21.5 GB of repository snapshots.
gemma-swe download core graphs embeddings wheels snapshots

# 4. Base model weights (W4A16 quantised; the 31B model is 23.3 GB on disk).
mkdir -p "data/models/$MODEL"
kaggle models instances versions download "google/gemma-4/other/$MODEL/$MODEL_VERSION" \
  -p "data/models/$MODEL" --untar

python - <<'PY'
import swegemma, adk_submission, vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GiB")
PY
echo "Setup complete. Next: scripts/gpu/run_eval.sh agents/baseline smoke --limit 3"
