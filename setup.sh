#!/usr/bin/env bash
# setup.sh -- One-time environment setup. Installs deps and pre-warms HF cache.
# Run inside the container/pod before any benchmarks.

source "$(dirname "$0")/config.sh"

echo "=== Setup: validating environment ==="

# Validate nvidia-smi
if ! nvidia-smi > /dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed. Are NVIDIA drivers and GPUs available?" >&2
  exit 1
fi
_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NR==1 {print; exit}')
echo "GPU check: OK ($_GPU_NAME)"

# Install system dependencies
echo "=== Setup: installing system packages ==="
apt-get update -qq
apt-get install -y -qq fio sysbench libaio-dev jq curl > /dev/null 2>&1
echo "System packages: OK"

# Install vLLM (optionally pinned via VLLM_VERSION env var, e.g. VLLM_VERSION=0.16.0)
echo "=== Setup: installing vllm ==="
if [[ -n "${VLLM_VERSION:-}" ]]; then
  pip install -q "vllm==${VLLM_VERSION}"
else
  pip install -q vllm
fi
_VLLM_INSTALLED=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: $_VLLM_INSTALLED"

# Pre-warm HuggingFace cache (download model weights)
echo "=== Setup: pre-warming HF cache for $MODEL ==="
python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL}',
    token=os.environ['HF_TOKEN'],
)
print('Model cache warm.')
"

echo ""
echo "=== Setup complete ==="
echo "vLLM: $_VLLM_INSTALLED"
echo "Model: $MODEL (cached)"
echo "Next: ./serve.sh & then ./run_all.sh --env-name <name>"
