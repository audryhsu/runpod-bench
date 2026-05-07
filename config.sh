#!/usr/bin/env bash
# config.sh -- Single source of truth for all benchmark parameters.
# Source this file from every script: source "$(dirname "$0")/config.sh"
# Override any value by exporting it before sourcing.

set -euo pipefail

# --- Required (no defaults) ---
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be set. Export it before running." >&2
  exit 1
fi
export HF_TOKEN

# --- Fixed values ---
export HF_HUB_DISABLE_XET=1
export SEED=42

# --- Model ---
export MODEL="${MODEL:-Qwen/Qwen3-8B}"

# --- vLLM server ---
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.95}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8128}"
export VLLM_DTYPE="${VLLM_DTYPE:-auto}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

# --- Workload ---
export INPUT_LEN="${INPUT_LEN:-512}"
export OUTPUT_LEN="${OUTPUT_LEN:-128}"
export NUM_PROMPTS="${NUM_PROMPTS:-200}"
export REQUEST_RATES="${REQUEST_RATES:-1 2 4 8 16 inf}"

# --- Repetitions ---
export BENCH_RUNS="${BENCH_RUNS:-5}"
export WARMUP_RUNS="${WARMUP_RUNS:-1}"

# --- fio ---
export FIO_DIR="${FIO_DIR:-/tmp/fio-bench}"
export FIO_SIZE="${FIO_SIZE:-10G}"
export FIO_RUNTIME="${FIO_RUNTIME:-30}"

# --- sysbench ---
export SYSBENCH_DURATION="${SYSBENCH_DURATION:-30}"

# --- Derived ---
export TOTAL_RUNS=$((WARMUP_RUNS + BENCH_RUNS))
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_URL="http://localhost:${VLLM_PORT}"
