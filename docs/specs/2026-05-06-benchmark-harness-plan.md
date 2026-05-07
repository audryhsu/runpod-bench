# Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable benchmark harness that compares vLLM inference performance in a vanilla Docker container vs a RunPod-managed pod on the same EC2 instance.

**Architecture:** Self-contained bash scripts + two Python scripts. All scripts source a shared `config.sh`. Each benchmark writes raw JSON to a timestamped results directory. `summarize.py` aggregates results; `compare.py` diffs two summaries. No environment-specific code paths.

**Tech Stack:** Bash, Python 3.11 (stdlib only -- no pip deps for analysis scripts), vLLM, fio, sysbench, jq

**Spec:** `docs/specs/2026-05-06-benchmark-harness-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `config.sh` | Single source of truth for all parameters. Sourced by every script. |
| `launch_baseline.sh` | Runs on EC2 host. `docker run` with correct flags for fair comparison. |
| `setup.sh` | One-time install of deps (fio, sysbench, jq, vllm). Pre-warms HF cache. |
| `capture_manifest.sh` | Fingerprints the environment (GPU, CPU, software versions, cgroup limits). |
| `serve.sh` | Starts vLLM server with consistent flags. Blocks until ready. |
| `bench_vllm_serving.sh` | Sweeps request rates, runs `vllm bench serve` N times per rate. |
| `bench_cold_start.sh` | Kills/restarts vLLM server, times readiness. |
| `bench_fio.sh` | Four fio profiles (seq read, seq write, rand read, rand write). |
| `bench_cpu.sh` | sysbench CPU + cgroup throttle snapshot. |
| `run_all.sh` | Top-level orchestrator. Calls all benchmarks in order, manages results dir. |
| `summarize.py` | Aggregates raw JSON into summary.json + summary.txt. |
| `compare.py` | Diffs two summaries, outputs delta table with color coding. |
| `.gitignore` | Ignores `results/` directory. |
| `README.md` | Usage instructions. |

---

### Task 1: config.sh + .gitignore

**Files:**
- Create: `config.sh`
- Create: `.gitignore`

- [ ] **Step 1: Create config.sh**

```bash
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
```

- [ ] **Step 2: Create .gitignore**

```
results/
*.pyc
__pycache__/
```

- [ ] **Step 3: Commit**

```bash
git add config.sh .gitignore
git commit -m "feat: add config.sh and .gitignore"
```

---

### Task 2: launch_baseline.sh

**Files:**
- Create: `launch_baseline.sh`

- [ ] **Step 1: Create launch_baseline.sh**

```bash
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
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x launch_baseline.sh
git add launch_baseline.sh
git commit -m "feat: add launch_baseline.sh for host-side container launch"
```

---

### Task 3: setup.sh

**Files:**
- Create: `setup.sh`

- [ ] **Step 1: Create setup.sh**

```bash
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
echo "GPU check: OK ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"

# Install system dependencies
echo "=== Setup: installing system packages ==="
apt-get update -qq
apt-get install -y -qq fio sysbench libaio-dev jq curl > /dev/null 2>&1
echo "System packages: OK"

# Install vLLM
echo "=== Setup: installing vllm ==="
pip install -q vllm
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
echo "vLLM version: $VLLM_VERSION"

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
echo "vLLM: $VLLM_VERSION"
echo "Model: $MODEL (cached)"
echo "Next: ./serve.sh & then ./run_all.sh --env-name <name>"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x setup.sh
git add setup.sh
git commit -m "feat: add setup.sh for dependency install and HF cache warmup"
```

---

### Task 4: capture_manifest.sh

**Files:**
- Create: `capture_manifest.sh`

- [ ] **Step 1: Create capture_manifest.sh**

```bash
#!/usr/bin/env bash
# capture_manifest.sh -- Emit manifest.json to stdout.
# Captures environment fingerprint for reproducibility and comparison.

source "$(dirname "$0")/config.sh"

# Detect container
IN_CONTAINER="false"
if [[ -f /.dockerenv ]] || [[ -f /run/.containerenv ]]; then
  IN_CONTAINER="true"
fi

# GPU info
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

# CPU info
CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
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
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x capture_manifest.sh
git add capture_manifest.sh
git commit -m "feat: add capture_manifest.sh for environment fingerprinting"
```

---

### Task 5: serve.sh

**Files:**
- Create: `serve.sh`

- [ ] **Step 1: Create serve.sh**

```bash
#!/usr/bin/env bash
# serve.sh -- Launch vLLM server with consistent flags.
# Blocks until the server is ready, then stays in foreground.
# Run in background: ./serve.sh &

source "$(dirname "$0")/config.sh"

ENFORCE_EAGER_FLAG=""
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  ENFORCE_EAGER_FLAG="--enforce-eager"
fi

echo "=== Starting vLLM server ==="
echo "Model: $MODEL"
echo "Port: $VLLM_PORT"
echo "GPU memory utilization: $VLLM_GPU_MEM_UTIL"
echo "Max model len: $VLLM_MAX_MODEL_LEN"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --dtype "$VLLM_DTYPE" \
  $ENFORCE_EAGER_FLAG \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --seed "$SEED"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x serve.sh
git add serve.sh
git commit -m "feat: add serve.sh for vLLM server launch"
```

---

### Task 6: bench_vllm_serving.sh

**Files:**
- Create: `bench_vllm_serving.sh`

- [ ] **Step 1: Create bench_vllm_serving.sh**

```bash
#!/usr/bin/env bash
# bench_vllm_serving.sh -- Sweep request rates with vllm bench serve.
# Usage: ./bench_vllm_serving.sh <results_dir>
# Expects vLLM server to be running already.

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"

# Verify server is reachable
if ! curl -sf "${VLLM_URL}/v1/models" > /dev/null 2>&1; then
  echo "ERROR: vLLM server not reachable at ${VLLM_URL}/v1/models" >&2
  echo "Start it with: ./serve.sh &" >&2
  exit 1
fi

echo "=== vLLM Serving Benchmark ==="
echo "Rates: $REQUEST_RATES"
echo "Runs per rate: $TOTAL_RUNS ($WARMUP_RUNS warmup + $BENCH_RUNS measured)"
echo "Prompts per run: $NUM_PROMPTS"
echo ""

for rate in $REQUEST_RATES; do
  RATE_DIR="${RESULTS_DIR}/vllm_serving/rate_${rate}"
  mkdir -p "$RATE_DIR"

  echo "--- Rate: $rate req/s ---"
  for run in $(seq 0 $((TOTAL_RUNS - 1))); do
    LABEL="run_${run}"
    if [[ $run -lt $WARMUP_RUNS ]]; then
      LABEL="${LABEL} (warmup)"
    fi
    echo "  $LABEL"

    RATE_FLAG="--request-rate $rate"
    if [[ "$rate" == "inf" ]]; then
      RATE_FLAG="--request-rate inf"
    fi

    vllm bench serve \
      --model "$MODEL" \
      --base-url "$VLLM_URL" \
      --endpoint /v1/completions \
      --num-prompts "$NUM_PROMPTS" \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      $RATE_FLAG \
      --save-result \
      --result-dir "$RATE_DIR" \
      --result-filename "run_${run}.json" \
      2>&1 | tee "${RATE_DIR}/run_${run}.log"

    echo ""
  done
done

echo "=== vLLM Serving Benchmark Complete ==="
echo "Results: ${RESULTS_DIR}/vllm_serving/"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x bench_vllm_serving.sh
git add bench_vllm_serving.sh
git commit -m "feat: add bench_vllm_serving.sh for request rate sweep"
```

---

### Task 7: bench_cold_start.sh

**Files:**
- Create: `bench_cold_start.sh`

- [ ] **Step 1: Create bench_cold_start.sh**

```bash
#!/usr/bin/env bash
# bench_cold_start.sh -- Measure vLLM cold start time (model load to ready).
# Usage: ./bench_cold_start.sh <results_dir>
# Will kill and restart the vLLM server for each run.

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"
COLD_DIR="${RESULTS_DIR}/cold_start"
mkdir -p "$COLD_DIR"

echo "=== Cold Start Benchmark ==="
echo "Runs: $TOTAL_RUNS ($WARMUP_RUNS warmup + $BENCH_RUNS measured)"
echo ""

kill_vllm() {
  # Kill any vllm serve processes
  pkill -f "vllm serve" 2>/dev/null || true
  # Wait for process to fully exit
  for i in $(seq 1 30); do
    if ! pgrep -f "vllm serve" > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  # Force kill if still running
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 2
}

wait_for_ready() {
  local max_wait=600  # 10 minutes max
  local elapsed=0
  while [[ $elapsed -lt $max_wait ]]; do
    if curl -sf "${VLLM_URL}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "ERROR: vLLM server did not become ready within ${max_wait}s" >&2
  return 1
}

for run in $(seq 0 $((TOTAL_RUNS - 1))); do
  LABEL="run_${run}"
  if [[ $run -lt $WARMUP_RUNS ]]; then
    LABEL="${LABEL} (warmup)"
  fi
  echo "--- $LABEL ---"

  # Kill existing server
  echo "  Stopping vLLM server..."
  kill_vllm

  # Drop page cache (best-effort)
  if sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
    echo "  Page cache dropped"
  else
    echo "  WARN: Could not drop page cache (needs root). Continuing."
  fi

  # Record start time and launch server
  START_EPOCH=$(date +%s.%N)
  echo "  Starting vLLM server..."
  "$SCRIPT_DIR/serve.sh" > "${COLD_DIR}/run_${run}_server.log" 2>&1 &
  SERVER_PID=$!

  # Wait for ready
  if wait_for_ready; then
    READY_EPOCH=$(date +%s.%N)
    DURATION=$(echo "$READY_EPOCH - $START_EPOCH" | bc)
    echo "  Ready in ${DURATION}s"

    jq -n \
      --argjson start_epoch "$START_EPOCH" \
      --argjson ready_epoch "$READY_EPOCH" \
      --argjson duration_s "$DURATION" \
      --argjson run "$run" \
      '{start_epoch:$start_epoch, ready_epoch:$ready_epoch, duration_s:$duration_s, run:$run}' \
      > "${COLD_DIR}/run_${run}.json"
  else
    echo "  FAIL: Server did not start"
    jq -n \
      --argjson start_epoch "$START_EPOCH" \
      --argjson run "$run" \
      '{start_epoch:$start_epoch, ready_epoch:null, duration_s:null, run:$run, error:"timeout"}' \
      > "${COLD_DIR}/run_${run}.json"
  fi

  echo ""
done

# Leave server running for subsequent benchmarks
echo "=== Cold Start Benchmark Complete ==="
echo "Server left running (PID: $SERVER_PID)"
echo "Results: $COLD_DIR/"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x bench_cold_start.sh
git add bench_cold_start.sh
git commit -m "feat: add bench_cold_start.sh for model load timing"
```

---

### Task 8: bench_fio.sh

**Files:**
- Create: `bench_fio.sh`

- [ ] **Step 1: Create bench_fio.sh**

```bash
#!/usr/bin/env bash
# bench_fio.sh -- Run 4 fio storage profiles.
# Usage: ./bench_fio.sh <results_dir>

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"
FIO_RESULTS="${RESULTS_DIR}/fio"
mkdir -p "$FIO_RESULTS" "$FIO_DIR"

echo "=== Storage Benchmark (fio) ==="
echo "Test dir: $FIO_DIR"
echo "File size: $FIO_SIZE"
echo "Runtime: ${FIO_RUNTIME}s per profile"
echo ""

run_fio() {
  local name="$1"
  local rw="$2"
  local bs="$3"
  local numjobs="$4"
  local iodepth="$5"
  local description="$6"

  echo "--- $name: $description ---"
  fio \
    --name="$name" \
    --rw="$rw" \
    --bs="$bs" \
    --size="$FIO_SIZE" \
    --numjobs="$numjobs" \
    --iodepth="$iodepth" \
    --direct=1 \
    --ioengine=libaio \
    --directory="$FIO_DIR" \
    --runtime="$FIO_RUNTIME" \
    --time_based \
    --group_reporting \
    --output-format=json \
    --output="${FIO_RESULTS}/${name}.json" \
    2>&1 | tee "${FIO_RESULTS}/${name}.log"

  echo "  -> ${FIO_RESULTS}/${name}.json"
  echo ""
}

run_fio "seq_read"   "read"      "1M"   1 32 "Sequential read (model loading)"
run_fio "seq_write"  "write"     "1M"   1 32 "Sequential write (checkpoint saving)"
run_fio "rand_read"  "randread"  "256k" 4 16 "Random read (dataset loading)"
run_fio "rand_write" "randwrite" "4k"   4 32 "Random write (docker layer extraction)"

# Cleanup test files
rm -f "${FIO_DIR}"/seq_read.* "${FIO_DIR}"/seq_write.* \
      "${FIO_DIR}"/rand_read.* "${FIO_DIR}"/rand_write.*

echo "=== Storage Benchmark Complete ==="
echo "Results: $FIO_RESULTS/"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x bench_fio.sh
git add bench_fio.sh
git commit -m "feat: add bench_fio.sh for storage throughput profiling"
```

---

### Task 9: bench_cpu.sh

**Files:**
- Create: `bench_cpu.sh`

- [ ] **Step 1: Create bench_cpu.sh**

```bash
#!/usr/bin/env bash
# bench_cpu.sh -- sysbench CPU benchmark + cgroup throttle snapshot.
# Usage: ./bench_cpu.sh <results_dir>

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"
CPU_DIR="${RESULTS_DIR}/cpu"
mkdir -p "$CPU_DIR"

THREADS=$(nproc)

echo "=== CPU Benchmark ==="
echo "Threads: $THREADS"
echo "Duration: ${SYSBENCH_DURATION}s"
echo ""

# --- sysbench ---
echo "--- sysbench cpu ---"
SYSBENCH_RAW=$(sysbench cpu --threads="$THREADS" --time="$SYSBENCH_DURATION" run 2>&1)
echo "$SYSBENCH_RAW" > "${CPU_DIR}/sysbench_raw.txt"

# Parse sysbench output
EVENTS_PER_SEC=$(echo "$SYSBENCH_RAW" | grep "events per second:" | awk '{print $NF}')
LAT_AVG=$(echo "$SYSBENCH_RAW" | grep "avg:" | awk '{print $NF}')
LAT_P95=$(echo "$SYSBENCH_RAW" | grep "95th percentile:" | awk '{print $NF}')
TOTAL_EVENTS=$(echo "$SYSBENCH_RAW" | grep "total number of events:" | awk '{print $NF}')

jq -n \
  --argjson events_per_sec "${EVENTS_PER_SEC:-0}" \
  --argjson latency_avg_ms "${LAT_AVG:-0}" \
  --argjson latency_p95_ms "${LAT_P95:-0}" \
  --argjson total_events "${TOTAL_EVENTS:-0}" \
  --argjson threads "$THREADS" \
  --argjson duration "$SYSBENCH_DURATION" \
  '{events_per_sec:$events_per_sec, latency_avg_ms:$latency_avg_ms, latency_p95_ms:$latency_p95_ms, total_events:$total_events, threads:$threads, duration_s:$duration}' \
  > "${CPU_DIR}/sysbench.json"

echo "  Events/sec: $EVENTS_PER_SEC"
echo "  Latency avg: ${LAT_AVG}ms, p95: ${LAT_P95}ms"
echo ""

echo "=== CPU Benchmark Complete ==="
echo "Results: $CPU_DIR/"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x bench_cpu.sh
git add bench_cpu.sh
git commit -m "feat: add bench_cpu.sh for CPU and cgroup throttle measurement"
```

---

### Task 10: run_all.sh

**Files:**
- Create: `run_all.sh`

This is the top-level orchestrator. It takes `--env-name` and `--runs`, creates the timestamped results directory, snapshots the cgroup state, runs all benchmarks in order, and calls summarize.py.

- [ ] **Step 1: Create run_all.sh**

```bash
#!/usr/bin/env bash
# run_all.sh -- Top-level orchestrator for all benchmarks.
# Usage: ./run_all.sh --env-name <name> [--runs <n>] [--skip-fio] [--skip-cpu] [--skip-cold-start]

source "$(dirname "$0")/config.sh"

# --- Parse arguments ---
ENV_NAME=""
SKIP_FIO=false
SKIP_CPU=false
SKIP_COLD_START=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)    ENV_NAME="$2"; shift 2 ;;
    --runs)        BENCH_RUNS="$2"; TOTAL_RUNS=$((WARMUP_RUNS + BENCH_RUNS)); shift 2 ;;
    --skip-fio)    SKIP_FIO=true; shift ;;
    --skip-cpu)    SKIP_CPU=true; shift ;;
    --skip-cold-start) SKIP_COLD_START=true; shift ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 --env-name <name> [--runs <n>] [--skip-fio] [--skip-cpu] [--skip-cold-start]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENV_NAME" ]]; then
  echo "ERROR: --env-name is required" >&2
  exit 1
fi

# --- Create results directory ---
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RESULTS_DIR="${SCRIPT_DIR}/results/${ENV_NAME}/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  RunPod Benchmark Harness"
echo "=============================================="
echo "Environment: $ENV_NAME"
echo "Timestamp:   $TIMESTAMP"
echo "Results:     $RESULTS_DIR"
echo "Runs:        $BENCH_RUNS (+ $WARMUP_RUNS warmup)"
echo "=============================================="
echo ""

# --- Capture manifest ---
echo ">>> Capturing environment manifest..."
"$SCRIPT_DIR/capture_manifest.sh" > "${RESULTS_DIR}/manifest.json"
echo "Manifest saved."
echo ""

# --- Snapshot cgroup state (before vLLM benchmark) ---
read_cgroup_cpu_stat() {
  local stat_file=""
  if [[ -f /sys/fs/cgroup/cpu.stat ]]; then
    stat_file="/sys/fs/cgroup/cpu.stat"
  elif [[ -f /sys/fs/cgroup/cpu/cpu.stat ]]; then
    stat_file="/sys/fs/cgroup/cpu/cpu.stat"
  fi

  if [[ -n "$stat_file" ]]; then
    local nr_throttled=$(grep "nr_throttled " "$stat_file" | awk '{print $2}')
    local throttled_usec=$(grep "throttled_usec " "$stat_file" | awk '{print $2}')
    # cgroup v1 uses throttled_time (nanoseconds), convert to usec
    if [[ -z "$throttled_usec" ]]; then
      local throttled_time=$(grep "throttled_time " "$stat_file" | awk '{print $2}')
      if [[ -n "$throttled_time" ]]; then
        throttled_usec=$((throttled_time / 1000))
      fi
    fi
    echo "${nr_throttled:-0} ${throttled_usec:-0}"
  else
    echo "null null"
  fi
}

CGROUP_BEFORE=$(read_cgroup_cpu_stat)
CGROUP_BEFORE_THROTTLED=$(echo "$CGROUP_BEFORE" | awk '{print $1}')
CGROUP_BEFORE_USEC=$(echo "$CGROUP_BEFORE" | awk '{print $2}')

# --- Tier 1: vLLM Serving Sweep ---
echo ">>> Tier 1: vLLM Serving Sweep"
"$SCRIPT_DIR/bench_vllm_serving.sh" "$RESULTS_DIR"
echo ""

# --- Snapshot cgroup state (after vLLM benchmark) ---
CGROUP_AFTER=$(read_cgroup_cpu_stat)
CGROUP_AFTER_THROTTLED=$(echo "$CGROUP_AFTER" | awk '{print $1}')
CGROUP_AFTER_USEC=$(echo "$CGROUP_AFTER" | awk '{print $2}')

# Write cgroup throttle delta
mkdir -p "${RESULTS_DIR}/cpu"
if [[ "$CGROUP_BEFORE_THROTTLED" == "null" ]]; then
  jq -n '{nr_throttled_delta:null, throttled_usec_delta:null, note:"cgroup cpu.stat not available"}' \
    > "${RESULTS_DIR}/cpu/cgroup_throttle.json"
  echo "WARN: cgroup cpu.stat not found -- throttle data unavailable"
else
  NR_DELTA=$((CGROUP_AFTER_THROTTLED - CGROUP_BEFORE_THROTTLED))
  USEC_DELTA=$((CGROUP_AFTER_USEC - CGROUP_BEFORE_USEC))
  jq -n \
    --argjson nr_throttled_delta "$NR_DELTA" \
    --argjson throttled_usec_delta "$USEC_DELTA" \
    --argjson before_nr "$CGROUP_BEFORE_THROTTLED" \
    --argjson after_nr "$CGROUP_AFTER_THROTTLED" \
    --argjson before_usec "$CGROUP_BEFORE_USEC" \
    --argjson after_usec "$CGROUP_AFTER_USEC" \
    '{nr_throttled_delta:$nr_throttled_delta, throttled_usec_delta:$throttled_usec_delta, before:{nr_throttled:$before_nr, throttled_usec:$before_usec}, after:{nr_throttled:$after_nr, throttled_usec:$after_usec}}' \
    > "${RESULTS_DIR}/cpu/cgroup_throttle.json"
  echo "Cgroup throttle delta: nr_throttled=$NR_DELTA, throttled_usec=$USEC_DELTA"
fi
echo ""

# --- Tier 2: Cold Start ---
if [[ "$SKIP_COLD_START" == "false" ]]; then
  echo ">>> Tier 2: Cold Start"
  "$SCRIPT_DIR/bench_cold_start.sh" "$RESULTS_DIR"
  echo ""
else
  echo ">>> Tier 2: Cold Start -- SKIPPED"
  echo ""
fi

# --- Tier 3: Storage (fio) ---
if [[ "$SKIP_FIO" == "false" ]]; then
  echo ">>> Tier 3: Storage (fio)"
  "$SCRIPT_DIR/bench_fio.sh" "$RESULTS_DIR"
  echo ""
else
  echo ">>> Tier 3: Storage (fio) -- SKIPPED"
  echo ""
fi

# --- Tier 4: CPU ---
if [[ "$SKIP_CPU" == "false" ]]; then
  echo ">>> Tier 4: CPU"
  "$SCRIPT_DIR/bench_cpu.sh" "$RESULTS_DIR"
  echo ""
else
  echo ">>> Tier 4: CPU -- SKIPPED"
  echo ""
fi

# --- Summarize ---
echo ">>> Generating summary..."
python "$SCRIPT_DIR/summarize.py" "$RESULTS_DIR"
echo ""

echo "=============================================="
echo "  Benchmark Complete"
echo "=============================================="
echo "Results:  $RESULTS_DIR"
echo "Summary:  ${RESULTS_DIR}/summary.txt"
echo ""
echo "To compare with another run:"
echo "  ./compare.py $RESULTS_DIR <other_results_dir>"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x run_all.sh
git add run_all.sh
git commit -m "feat: add run_all.sh orchestrator with cgroup snapshots and skip flags"
```

---

### Task 11: summarize.py

**Files:**
- Create: `summarize.py`

This uses only Python stdlib (json, pathlib, statistics). No pip deps.

- [ ] **Step 1: Create summarize.py**

```python
#!/usr/bin/env python3
"""summarize.py -- Aggregate raw benchmark JSON into summary.json + summary.txt.

Usage: python summarize.py <results_dir>

The results_dir should contain subdirectories: vllm_serving/, cold_start/, fio/, cpu/
"""

import json
import sys
import os
from pathlib import Path
from statistics import median


def percentile(data, p):
    """Compute p-th percentile (0-100) of a sorted list."""
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def load_json(path):
    """Load JSON file, return None if missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARN: Could not load {path}: {e}", file=sys.stderr)
        return None


def summarize_vllm_serving(results_dir, warmup_runs):
    """Summarize vLLM serving benchmark results."""
    serving_dir = results_dir / "vllm_serving"
    if not serving_dir.exists():
        return None

    summary = {}
    for rate_dir in sorted(serving_dir.iterdir()):
        if not rate_dir.is_dir() or not rate_dir.name.startswith("rate_"):
            continue
        rate = rate_dir.name.replace("rate_", "")

        runs = []
        for run_file in sorted(rate_dir.glob("run_*.json")):
            run_num = int(run_file.stem.split("_")[1])
            if run_num < warmup_runs:
                continue  # skip warmup
            data = load_json(run_file)
            if data is not None:
                runs.append(data)

        if not runs:
            continue

        # Extract metrics from vllm bench serve output
        # vllm bench serve --save-result writes a JSON with various metrics
        metrics = {}
        for key in [
            "request_throughput",
            "output_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "p99_itl_ms",
        ]:
            values = [r[key] for r in runs if key in r]
            if values:
                metrics[key] = {
                    "median": round(median(values), 3),
                    "p95": round(percentile(values, 95), 3),
                    "values": [round(v, 3) for v in values],
                }

        summary[rate] = metrics

    return summary if summary else None


def summarize_cold_start(results_dir, warmup_runs):
    """Summarize cold start benchmark results."""
    cold_dir = results_dir / "cold_start"
    if not cold_dir.exists():
        return None

    durations = []
    for run_file in sorted(cold_dir.glob("run_*.json")):
        run_num = int(run_file.stem.split("_")[1])
        if run_num < warmup_runs:
            continue
        data = load_json(run_file)
        if data and data.get("duration_s") is not None:
            durations.append(data["duration_s"])

    if not durations:
        return None

    return {
        "median_s": round(median(durations), 2),
        "p95_s": round(percentile(durations, 95), 2),
        "values": [round(d, 2) for d in durations],
    }


def summarize_fio(results_dir):
    """Summarize fio benchmark results."""
    fio_dir = results_dir / "fio"
    if not fio_dir.exists():
        return None

    summary = {}
    for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
        data = load_json(fio_dir / f"{profile}.json")
        if data is None:
            continue

        jobs = data.get("jobs", [{}])
        if not jobs:
            continue
        job = jobs[0]

        # Determine read or write based on profile
        if "read" in profile:
            io = job.get("read", {})
        else:
            io = job.get("write", {})

        bw_bytes = io.get("bw_bytes", io.get("bw", 0) * 1024)
        bw_mbs = round(bw_bytes / (1024 * 1024), 1)
        iops = round(io.get("iops", 0), 1)
        lat_ns = io.get("clat_ns", io.get("lat_ns", {}))
        lat_p50_ms = round(lat_ns.get("percentile", {}).get("50.000000", 0) / 1e6, 3)
        lat_p99_ms = round(lat_ns.get("percentile", {}).get("99.000000", 0) / 1e6, 3)

        summary[profile] = {
            "bw_mbs": bw_mbs,
            "iops": iops,
            "lat_p50_ms": lat_p50_ms,
            "lat_p99_ms": lat_p99_ms,
        }

    return summary if summary else None


def summarize_cpu(results_dir):
    """Summarize CPU benchmark results."""
    cpu_dir = results_dir / "cpu"
    if not cpu_dir.exists():
        return None

    summary = {}

    sysbench = load_json(cpu_dir / "sysbench.json")
    if sysbench:
        summary["sysbench"] = sysbench

    cgroup = load_json(cpu_dir / "cgroup_throttle.json")
    if cgroup:
        summary["cgroup_throttle"] = cgroup

    return summary if summary else None


def format_summary_txt(summary):
    """Format summary as human-readable text."""
    lines = []

    # vLLM Serving
    vllm = summary.get("vllm_serving")
    if vllm:
        lines.append("=== vLLM Serving ===")
        lines.append(
            f"{'Rate':<6} | {'Tput req/s':>10} | {'Tput tok/s':>10} | {'TTFT p50':>10} | {'TTFT p99':>10} | {'TPOT p50':>10}"
        )
        lines.append("-" * 75)
        for rate in sorted(vllm.keys(), key=lambda r: float(r) if r != "inf" else float("inf")):
            m = vllm[rate]
            req_s = m.get("request_throughput", {}).get("median", "-")
            tok_s = m.get("output_throughput", {}).get("median", "-")
            ttft_p50 = m.get("median_ttft_ms", {}).get("median", "-")
            ttft_p99 = m.get("p99_ttft_ms", {}).get("median", "-")
            tpot_p50 = m.get("median_tpot_ms", {}).get("median", "-")

            def fmt(v):
                return f"{v}" if v == "-" else f"{v:.1f}"

            lines.append(
                f"{rate:<6} | {fmt(req_s):>10} | {fmt(tok_s):>10} | {fmt(ttft_p50):>10} | {fmt(ttft_p99):>10} | {fmt(tpot_p50):>10}"
            )
        lines.append("")

    # Cold Start
    cold = summary.get("cold_start")
    if cold:
        lines.append("=== Cold Start ===")
        lines.append(f"Median: {cold['median_s']}s | p95: {cold['p95_s']}s")
        lines.append("")

    # FIO
    fio = summary.get("fio")
    if fio:
        lines.append("=== Storage (fio) ===")
        lines.append(
            f"{'Profile':<12} | {'BW (MB/s)':>10} | {'IOPS':>10} | {'Lat p50':>10} | {'Lat p99':>10}"
        )
        lines.append("-" * 60)
        for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
            if profile in fio:
                m = fio[profile]
                lines.append(
                    f"{profile:<12} | {m['bw_mbs']:>10.1f} | {m['iops']:>10.1f} | {m['lat_p50_ms']:>8.3f}ms | {m['lat_p99_ms']:>8.3f}ms"
                )
        lines.append("")

    # CPU
    cpu = summary.get("cpu")
    if cpu:
        lines.append("=== CPU ===")
        sb = cpu.get("sysbench", {})
        if sb:
            lines.append(
                f"sysbench: {sb.get('events_per_sec', '-')} events/s, "
                f"avg lat {sb.get('latency_avg_ms', '-')}ms, "
                f"p95 lat {sb.get('latency_p95_ms', '-')}ms"
            )
        cg = cpu.get("cgroup_throttle", {})
        if cg:
            nr = cg.get("nr_throttled_delta", "n/a")
            usec = cg.get("throttled_usec_delta", "n/a")
            lines.append(f"cgroup throttle: nr_throttled={nr}, throttled_usec={usec}")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load config from manifest
    manifest = load_json(results_dir / "manifest.json")
    warmup_runs = 1
    if manifest and "config_snapshot" in manifest:
        warmup_runs = int(manifest["config_snapshot"].get("warmup_runs", 1))

    summary = {}

    print("Summarizing vLLM serving...")
    vllm = summarize_vllm_serving(results_dir, warmup_runs)
    if vllm:
        summary["vllm_serving"] = vllm

    print("Summarizing cold start...")
    cold = summarize_cold_start(results_dir, warmup_runs)
    if cold:
        summary["cold_start"] = cold

    print("Summarizing fio...")
    fio = summarize_fio(results_dir)
    if fio:
        summary["fio"] = fio

    print("Summarizing CPU...")
    cpu = summarize_cpu(results_dir)
    if cpu:
        summary["cpu"] = cpu

    # Write summary.json
    summary_json_path = results_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_json_path}")

    # Write summary.txt
    txt = format_summary_txt(summary)
    summary_txt_path = results_dir / "summary.txt"
    with open(summary_txt_path, "w") as f:
        f.write(txt)
    print(f"Wrote {summary_txt_path}")
    print()
    print(txt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x summarize.py
git add summarize.py
git commit -m "feat: add summarize.py for result aggregation"
```

---

### Task 12: compare.py

**Files:**
- Create: `compare.py`

- [ ] **Step 1: Create compare.py**

```python
#!/usr/bin/env python3
"""compare.py -- Diff two benchmark summaries and output a delta table.

Usage: python compare.py <baseline_results_dir> <treatment_results_dir>

Outputs a table showing baseline vs treatment with percent deltas.
Color-coded: green (<5% delta), yellow (5-15%), red (>15%).
"""

import json
import sys
from pathlib import Path


# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARN: Could not load {path}: {e}", file=sys.stderr)
        return None


def color_delta(delta_pct, higher_is_worse=True):
    """Color a delta percentage. For latency, higher is worse. For throughput, lower is worse."""
    if delta_pct is None:
        return "n/a"
    abs_delta = abs(delta_pct)
    # Determine if this delta is bad
    is_bad = (delta_pct > 0 and higher_is_worse) or (delta_pct < 0 and not higher_is_worse)

    if abs_delta < 5:
        color = GREEN
    elif abs_delta < 15:
        color = YELLOW
    else:
        color = RED if is_bad else GREEN

    sign = "+" if delta_pct > 0 else ""
    return f"{color}{sign}{delta_pct:.1f}%{RESET}"


def pct_delta(baseline, treatment):
    """Compute percent delta from baseline to treatment."""
    if baseline is None or treatment is None or baseline == 0:
        return None
    return ((treatment - baseline) / abs(baseline)) * 100


def print_manifest_comparison(m1, m2, name1, name2):
    """Print side-by-side manifest comparison."""
    print(f"\n{BOLD}=== Manifest Comparison ==={RESET}")
    fields = [
        "hostname", "in_container", "gpu_model", "gpu_count", "gpu_driver",
        "cuda_version", "cpu_model", "cpu_count", "memory_gb",
        "vllm_version", "torch_version", "cgroup_cpu_quota", "cgroup_memory_limit",
    ]
    print(f"{'':>22} | {name1:<22} | {name2:<22}")
    print("-" * 72)
    for field in fields:
        v1 = str(m1.get(field, "n/a"))[:22]
        v2 = str(m2.get(field, "n/a"))[:22]
        marker = " *" if v1 != v2 else ""
        print(f"{field:>22} | {v1:<22} | {v2:<22}{marker}")

    # Warnings
    warnings = []
    if m1.get("gpu_model") != m2.get("gpu_model"):
        warnings.append("WARNING: GPU models differ!")
    if m1.get("gpu_count") != m2.get("gpu_count"):
        warnings.append("WARNING: GPU counts differ!")
    if m1.get("vllm_version") != m2.get("vllm_version"):
        warnings.append("WARNING: vLLM versions differ!")
    for w in warnings:
        print(f"\n{RED}{w}{RESET}")


def print_vllm_comparison(s1, s2, name1, name2):
    """Print vLLM serving comparison table."""
    if not s1 or not s2:
        print("\n(vLLM serving data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== vLLM Serving ==={RESET}")

    metrics = [
        ("request_throughput",  "tput req/s",  False),  # higher is better
        ("output_throughput",   "tput tok/s",  False),
        ("median_ttft_ms",      "TTFT p50",    True),   # higher is worse
        ("p99_ttft_ms",         "TTFT p99",    True),
        ("median_tpot_ms",      "TPOT p50",    True),
        ("median_itl_ms",       "ITL p50",     True),
    ]

    print(f"{'Rate':<6} | {'Metric':<12} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 65)

    all_rates = sorted(
        set(list(s1.keys()) + list(s2.keys())),
        key=lambda r: float(r) if r != "inf" else float("inf"),
    )

    for rate in all_rates:
        r1 = s1.get(rate, {})
        r2 = s2.get(rate, {})
        for key, label, higher_is_worse in metrics:
            v1 = r1.get(key, {}).get("median")
            v2 = r2.get(key, {}).get("median")
            delta = pct_delta(v1, v2)
            delta_str = color_delta(delta, higher_is_worse)

            v1_str = f"{v1:.1f}" if v1 is not None else "n/a"
            v2_str = f"{v2:.1f}" if v2 is not None else "n/a"
            print(f"{rate:<6} | {label:<12} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")
        print("-" * 65)


def print_cold_start_comparison(s1, s2, name1, name2):
    """Print cold start comparison."""
    if not s1 or not s2:
        print("\n(Cold start data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== Cold Start ==={RESET}")
    print(f"{'Metric':<8} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 55)

    for key, label in [("median_s", "median"), ("p95_s", "p95")]:
        v1 = s1.get(key)
        v2 = s2.get(key)
        delta = pct_delta(v1, v2)
        delta_str = color_delta(delta, higher_is_worse=True)
        v1_str = f"{v1:.1f}s" if v1 is not None else "n/a"
        v2_str = f"{v2:.1f}s" if v2 is not None else "n/a"
        print(f"{label:<8} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")


def print_fio_comparison(s1, s2, name1, name2):
    """Print fio comparison."""
    if not s1 or not s2:
        print("\n(FIO data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== Storage (fio) ==={RESET}")
    print(f"{'Profile':<12} | {'Metric':<10} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 68)

    for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
        p1 = s1.get(profile, {})
        p2 = s2.get(profile, {})
        for key, label, higher_is_worse in [
            ("bw_mbs", "BW MB/s", False),
            ("iops", "IOPS", False),
            ("lat_p50_ms", "Lat p50", True),
            ("lat_p99_ms", "Lat p99", True),
        ]:
            v1 = p1.get(key)
            v2 = p2.get(key)
            delta = pct_delta(v1, v2)
            delta_str = color_delta(delta, higher_is_worse)
            v1_str = f"{v1:.1f}" if v1 is not None else "n/a"
            v2_str = f"{v2:.1f}" if v2 is not None else "n/a"
            print(f"{profile:<12} | {label:<10} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")
        print("-" * 68)


def print_cpu_comparison(s1, s2, name1, name2):
    """Print CPU comparison."""
    if not s1 or not s2:
        print("\n(CPU data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== CPU ==={RESET}")
    print(f"{'Metric':<20} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 62)

    # sysbench
    sb1 = s1.get("sysbench", {})
    sb2 = s2.get("sysbench", {})
    for key, label, higher_is_worse in [
        ("events_per_sec", "events/s", False),
        ("latency_avg_ms", "lat avg ms", True),
        ("latency_p95_ms", "lat p95 ms", True),
    ]:
        v1 = sb1.get(key)
        v2 = sb2.get(key)
        delta = pct_delta(v1, v2)
        delta_str = color_delta(delta, higher_is_worse)
        v1_str = f"{v1:.1f}" if v1 is not None else "n/a"
        v2_str = f"{v2:.1f}" if v2 is not None else "n/a"
        print(f"{label:<20} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")

    # cgroup throttle
    cg1 = s1.get("cgroup_throttle", {})
    cg2 = s2.get("cgroup_throttle", {})
    for key, label in [("nr_throttled_delta", "throttled count"), ("throttled_usec_delta", "throttled usec")]:
        v1 = cg1.get(key)
        v2 = cg2.get(key)
        if v1 is None and v2 is None:
            continue
        v1_str = str(v1) if v1 is not None else "n/a"
        v2_str = str(v2) if v2 is not None else "n/a"
        # Absolute delta for throttle (not percent -- baseline is often 0)
        if v1 is not None and v2 is not None:
            abs_delta = v2 - v1
            if abs_delta == 0:
                delta_str = f"{GREEN}0{RESET}"
            else:
                delta_str = f"{RED}{'+' if abs_delta > 0 else ''}{abs_delta}{RESET}"
        else:
            delta_str = "n/a"
        print(f"{label:<20} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline_dir> <treatment_dir>", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    treat_dir = Path(sys.argv[2])

    # Derive names from directory structure: results/<env-name>/<timestamp>
    base_name = base_dir.parent.name
    treat_name = treat_dir.parent.name

    # Load data
    base_manifest = load_json(base_dir / "manifest.json") or {}
    treat_manifest = load_json(treat_dir / "manifest.json") or {}
    base_summary = load_json(base_dir / "summary.json") or {}
    treat_summary = load_json(treat_dir / "summary.json") or {}

    # Print comparison
    print(f"\n{BOLD}RunPod Benchmark Comparison{RESET}")
    print(f"Baseline:  {base_dir}")
    print(f"Treatment: {treat_dir}")

    print_manifest_comparison(base_manifest, treat_manifest, base_name, treat_name)
    print_vllm_comparison(
        base_summary.get("vllm_serving"),
        treat_summary.get("vllm_serving"),
        base_name, treat_name,
    )
    print_cold_start_comparison(
        base_summary.get("cold_start"),
        treat_summary.get("cold_start"),
        base_name, treat_name,
    )
    print_fio_comparison(
        base_summary.get("fio"),
        treat_summary.get("fio"),
        base_name, treat_name,
    )
    print_cpu_comparison(
        base_summary.get("cpu"),
        treat_summary.get("cpu"),
        base_name, treat_name,
    )
    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x compare.py
git add compare.py
git commit -m "feat: add compare.py for A/B delta table with color coding"
```

---

### Task 13: README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# runpod-bench

Benchmark harness comparing vLLM inference performance in a vanilla Docker container vs a RunPod-managed pod on the same EC2 instance. Measures orchestration overhead for RunPod Everywhere.

## Quick Start

### Prerequisites

- EC2 GPU instance with Docker and NVIDIA drivers installed
- HuggingFace token with model access
- The same Docker image available in both environments:
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

### Baseline (A): Vanilla Docker on EC2

**Important:** Ensure no RunPod pod is running on the machine first.

```bash
# On the EC2 host
cd runpod-bench
export HF_TOKEN=hf_xxxxx
./launch_baseline.sh

# Inside the container:
./setup.sh
./serve.sh &
./run_all.sh --env-name ec2-direct --runs 5
exit
```

### Treatment (B): RunPod Pod on EC2

```bash
# Deploy pod via dashboard/GraphQL, then:
# runpodctl exec into the pod
# Copy harness into the pod

export HF_TOKEN=hf_xxxxx
./setup.sh
./serve.sh &
./run_all.sh --env-name runpod-pod --runs 5
```

### Compare

```bash
# On the EC2 host (both results dirs accessible)
./compare.py results/ec2-direct/<timestamp> results/runpod-pod/<timestamp>
```

## Benchmarks

| Tier | Benchmark | Tool | Measures |
|------|-----------|------|----------|
| 1 | vLLM serving sweep | `vllm bench serve` | Inference throughput/latency at various request rates |
| 2 | Cold start | time `vllm serve` to ready | Model load time (disk to GPU) |
| 3 | Storage | `fio` | Raw disk I/O (seq read/write, random read/write) |
| 4 | CPU | `sysbench` + cgroup stats | CPU throughput and container throttling |

## Configuration

Edit `config.sh` or export env vars before sourcing. Key parameters:

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL | Qwen/Qwen3-8B | Model to benchmark |
| BENCH_RUNS | 5 | Measurement runs per benchmark |
| REQUEST_RATES | "1 2 4 8 16 inf" | Request rates for vLLM sweep |
| NUM_PROMPTS | 200 | Prompts per vLLM run |

## Skip flags

```bash
./run_all.sh --env-name test --skip-fio --skip-cpu --skip-cold-start
```

## Best Practices

- **Run one environment at a time.** Never run baseline and treatment simultaneously.
- **Interleave when possible.** For best results: A run 1, B run 1, A run 2, B run 2, etc.
  (requires manual orchestration -- the harness runs all N runs sequentially)
- Warmup runs are automatically discarded.
- Seeds are pinned for reproducibility.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with usage instructions"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - config.sh: Task 1 -- all parameters from spec
  - launch_baseline.sh: Task 2 -- Docker flags matched to RunPod pod config
  - setup.sh: Task 3 -- deps, HF cache warmup, validation
  - capture_manifest.sh: Task 4 -- all manifest fields from spec
  - serve.sh: Task 5 -- consistent vLLM flags
  - bench_vllm_serving.sh: Task 6 -- request rate sweep with warmup
  - bench_cold_start.sh: Task 7 -- kill/restart/poll cycle
  - bench_fio.sh: Task 8 -- 4 profiles from spec
  - bench_cpu.sh: Task 9 -- sysbench + cgroup
  - run_all.sh: Task 10 -- orchestrator with cgroup snapshots
  - summarize.py: Task 11 -- aggregation with warmup discard
  - compare.py: Task 12 -- delta table with color coding and warnings
  - README.md: Task 13 -- usage instructions
- **No placeholders:** All code is complete. No TBDs.
- **Type consistency:** JSON field names match across bench scripts, summarize.py, and compare.py.
- **Sequential-only constraint:** Documented in spec and README.
