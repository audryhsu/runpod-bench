#!/usr/bin/env bash
# launch_baseline.sh -- Run on the EC2 HOST (not inside a container).
# Launches a Docker container with flags matched to RunPod pod configuration
# so the benchmark comparison is fair.
#
# Usage: export HF_TOKEN=hf_xxx && ./launch_baseline.sh

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be set. Export it before running." >&2
  exit 1
fi

IMAGE="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Ensure HF cache directory exists on host
mkdir -p "$HF_CACHE"

echo "=== Launching baseline container ==="
echo "Image: $IMAGE"
echo "HF cache: $HF_CACHE"
echo "Harness mount: $(pwd) -> /bench"
echo ""
echo "You will be dropped into a bash shell inside the container."
echo "Run: ./setup.sh && ./serve.sh & && ./run_all.sh --env-name ec2-direct"
echo ""

docker run --gpus all -it --rm \
  --shm-size=16g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd)":/bench \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -w /bench \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HUB_DISABLE_XET=1 \
  "$IMAGE" \
  bash
