#!/usr/bin/env bash
# capture_manifest.sh -- Emit manifest.json to stdout.
# Captures environment fingerprint for reproducibility and comparison.

source "$(dirname "$0")/config.sh"

# Detect container
IN_CONTAINER="false"
if [[ -f /.dockerenv ]] || [[ -f /run/.containerenv ]]; then
  IN_CONTAINER="true"
fi

# GPU info -- avoid `... | head -1` (SIGPIPE under pipefail).
# Capture full output once, then parse with awk.
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
GPU_DRIVERS=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "")
GPU_CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "")
GPU_MODEL=$(echo "$GPU_NAMES" | awk 'NR==1 {sub(/^[ \t]+/,""); sub(/[ \t]+$/,""); print; exit}')
GPU_COUNT=$(echo "$GPU_NAMES" | awk 'NF{c++} END{print c+0}')
GPU_DRIVER=$(echo "$GPU_DRIVERS" | awk 'NR==1 {sub(/^[ \t]+/,""); sub(/[ \t]+$/,""); print; exit}')
CUDA_VERSION=$(echo "$GPU_CAPS" | awk 'NR==1 {sub(/^[ \t]+/,""); sub(/[ \t]+$/,""); print; exit}')
[[ -z "$CUDA_VERSION" ]] && CUDA_VERSION="unknown"

# CPU info
CPU_MODEL=$(awk -F: '/^model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)
CPU_COUNT=$(nproc)

# Memory
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')

# Software versions
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "not installed")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')

# OS
KERNEL=$(uname -r)
DISTRO=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"' || echo "unknown")

# cgroup limits (prefer v2, fallback v1)
CGROUP_CPU_QUOTA="unknown"
CGROUP_MEM_LIMIT="unknown"
if [[ -f /sys/fs/cgroup/cpu.max ]]; then
  CGROUP_CPU_QUOTA=$(cat /sys/fs/cgroup/cpu.max)
elif [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  CGROUP_CPU_QUOTA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
fi
if [[ -f /sys/fs/cgroup/memory.max ]]; then
  CGROUP_MEM_LIMIT=$(cat /sys/fs/cgroup/memory.max)
elif [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  CGROUP_MEM_LIMIT=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
fi

# Config snapshot
CONFIG_SNAPSHOT=$(jq -n \
  --arg model "$MODEL" \
  --arg vllm_port "$VLLM_PORT" \
  --arg gpu_mem_util "$VLLM_GPU_MEM_UTIL" \
  --arg max_model_len "$VLLM_MAX_MODEL_LEN" \
  --arg dtype "$VLLM_DTYPE" \
  --arg enforce_eager "$VLLM_ENFORCE_EAGER" \
  --arg input_len "$INPUT_LEN" \
  --arg output_len "$OUTPUT_LEN" \
  --arg num_prompts "$NUM_PROMPTS" \
  --arg request_rates "$REQUEST_RATES" \
  --arg bench_runs "$BENCH_RUNS" \
  --arg warmup_runs "$WARMUP_RUNS" \
  --arg seed "$SEED" \
  '{model:$model,vllm_port:$vllm_port,gpu_mem_util:$gpu_mem_util,max_model_len:$max_model_len,dtype:$dtype,enforce_eager:$enforce_eager,input_len:$input_len,output_len:$output_len,num_prompts:$num_prompts,request_rates:$request_rates,bench_runs:$bench_runs,warmup_runs:$warmup_runs,seed:$seed}')

# Emit JSON
jq -n \
  --arg hostname "$(hostname)" \
  --argjson in_container "$IN_CONTAINER" \
  --arg kernel "$KERNEL" \
  --arg distro "$DISTRO" \
  --arg gpu_model "$GPU_MODEL" \
  --argjson gpu_count "$GPU_COUNT" \
  --arg gpu_driver "$GPU_DRIVER" \
  --arg cuda_version "$CUDA_VERSION" \
  --arg cpu_model "$CPU_MODEL" \
  --argjson cpu_count "$CPU_COUNT" \
  --argjson memory_gb "$MEMORY_GB" \
  --arg vllm_version "$VLLM_VERSION" \
  --arg torch_version "$TORCH_VERSION" \
  --arg python_version "$PYTHON_VERSION" \
  --arg cgroup_cpu_quota "$CGROUP_CPU_QUOTA" \
  --arg cgroup_memory_limit "$CGROUP_MEM_LIMIT" \
  --argjson config_snapshot "$CONFIG_SNAPSHOT" \
  --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{
    hostname: $hostname,
    in_container: $in_container,
    kernel: $kernel,
    distro: $distro,
    gpu_model: $gpu_model,
    gpu_count: $gpu_count,
    gpu_driver: $gpu_driver,
    cuda_version: $cuda_version,
    cpu_model: $cpu_model,
    cpu_count: $cpu_count,
    memory_gb: $memory_gb,
    vllm_version: $vllm_version,
    torch_version: $torch_version,
    python_version: $python_version,
    cgroup_cpu_quota: $cgroup_cpu_quota,
    cgroup_memory_limit: $cgroup_memory_limit,
    config_snapshot: $config_snapshot,
    timestamp: $timestamp
  }'
