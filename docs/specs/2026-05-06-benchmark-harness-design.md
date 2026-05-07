# Benchmark Harness Design: RunPod-on-EC2 vs Direct EC2

**Date:** 2026-05-06
**Status:** Draft
**Author:** Audry Hsu

## Purpose

Quantify the performance overhead of running an LLM inference workload inside a RunPod pod hosted on an EC2 instance, versus running the same workload in a vanilla Docker container on the same EC2 instance. The output must be credible enough to share with PMs and prospective RPE customers: "if you bring your EC2 to RunPod, here's what it costs you in performance."

## Comparison Model

Both environments are containerized using the same Docker image (`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`). The only variable is RunPod's orchestration layer (cgroup policies, overlay networking, volume mounts, port proxying). This matches the real customer decision: "should I self-manage Docker on my EC2, or use RunPod to manage it?"

| | Baseline (A): ec2-direct | Treatment (B): runpod-pod |
|---|---|---|
| Runtime | `docker run --gpus all` on bare EC2 | RunPod pod on same EC2 instance |
| Image | runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 | same |
| Instance | g6.48xlarge (8x L4) | same physical instance |
| Client | Inside container, loopback | Inside pod, loopback |

## Execution Model

The harness is a self-contained directory of bash scripts + two Python scripts. It is environment-agnostic -- no code paths that branch on "am I in a RunPod pod." You copy it into whatever environment you're benchmarking, run it, and it produces structured JSON output.

### Operator Workflow

**Important: only one environment at a time.** The baseline container and RunPod pod must not run simultaneously. GPU memory (vLLM claims 95% of VRAM), CPU, and IO bandwidth are shared resources -- running both would measure contention, not orchestration overhead. Stop/remove one before starting the other.

```
# --- Baseline (A): vanilla Docker on EC2 ---
# Ensure no RunPod pod is running on this machine (stop it from dashboard or GraphQL)
# SSH into EC2 host, from the runpod-bench/ directory:
export HF_TOKEN=hf_xxxxx
./launch_baseline.sh        # docker run with correct flags, drops you into container

# Inside container (auto-lands in /bench):
./setup.sh
./serve.sh &                # or in tmux
./run_all.sh --env-name ec2-direct --runs 5
exit                        # results are on the host via bind mount

# --- Treatment (B): RunPod pod on same EC2 ---
# Deploy pod via GraphQL or dashboard
# runpodctl exec into pod, copy harness in (scp or git clone)
cd /bench
export HF_TOKEN=hf_xxxxx
./setup.sh
./serve.sh &
./run_all.sh --env-name runpod-pod --runs 5

# --- Compare (on EC2 host) ---
./compare.py results/ec2-direct/<ts> results/runpod-pod/<ts>
```

### launch_baseline.sh (runs on EC2 host, not inside container)

Launches the baseline Docker container with flags matched to what RunPod configures for pods. This ensures the comparison isolates RunPod's orchestration overhead, not differences in Docker run flags.

```bash
docker run --gpus all -it --rm \
  --shm-size=16g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd)":/bench \
  -w /bench \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HUB_DISABLE_XET=1 \
  runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  bash
```

Key flags:
- `--shm-size=16g` -- RunPod pods get enlarged shared memory; without this, PyTorch/vLLM multi-GPU can fail or degrade
- `--ipc=host` -- matches RunPod's IPC namespace sharing for NCCL
- `--ulimit memlock=-1` -- unlimited locked memory, required for GPU DMA
- `-v $(pwd):/bench` -- bind-mounts the harness + results directory so data persists after container exit

## Configuration

All tunable parameters live in `config.sh`. Every script sources it first. Values are env vars with defaults, overridable by exporting before sourcing.

### Fixed values (not configurable)

| Parameter | Value | Rationale |
|---|---|---|
| HF_HUB_DISABLE_XET | 1 | Xet protocol hangs on EC2 Docker bridge networks |
| seed | 42 | Reproducibility across runs |

### Configurable with defaults

| Parameter | Default | Description |
|---|---|---|
| MODEL | Qwen/Qwen3-8B | HuggingFace model ID |
| HF_TOKEN | (required, no default) | HuggingFace access token |
| VLLM_PORT | 8000 | vLLM server port |
| VLLM_GPU_MEM_UTIL | 0.95 | --gpu-memory-utilization |
| VLLM_MAX_MODEL_LEN | 8128 | --max-model-len |
| VLLM_DTYPE | auto | --dtype |
| VLLM_ENFORCE_EAGER | 1 | --enforce-eager (set to 0 to disable) |
| INPUT_LEN | 512 | Prompt input token length |
| OUTPUT_LEN | 128 | Max output tokens |
| NUM_PROMPTS | 200 | Prompts per benchmark run |
| REQUEST_RATES | "1 2 4 8 16 inf" | Space-separated request rates for sweep |
| BENCH_RUNS | 5 | Measurement runs (excluding warmup) |
| WARMUP_RUNS | 1 | Warmup runs discarded before measurement |
| FIO_DIR | /tmp/fio-bench | fio test directory |
| FIO_SIZE | 10G | fio test file size |
| FIO_RUNTIME | 30 | fio test duration (seconds) |
| SYSBENCH_DURATION | 30 | sysbench cpu duration (seconds) |

## Benchmarks

### Tier 1: vLLM Serving Sweep (bench_vllm_serving.sh)

Primary product signal. Sweeps request rates to find the throughput/latency curve.

For each rate in REQUEST_RATES, for each run in 0..(WARMUP_RUNS + BENCH_RUNS - 1):
1. Call `vllm bench serve` with pinned seed, input/output lengths, num prompts, and the target request rate
2. Save raw JSON output to `results/<env>/<ts>/vllm_serving/rate_<rate>/run_<n>.json`
3. Run indices 0..(WARMUP_RUNS-1) are warmup, discarded by summarize.py

Key metrics extracted per run: request throughput (req/s), token throughput (tok/s), TTFT p50/p95/p99, TPOT p50/p95, ITL p50/p95.

### Tier 2: Cold Start (bench_cold_start.sh)

Customer-visible scale-out latency. Measures how long vLLM takes to load the model and become ready to serve.

For each run:
1. Kill the vLLM server process (SIGTERM, wait for exit)
2. Drop page cache: `sync; echo 3 > /proc/sys/vm/drop_caches` (best-effort, requires root -- skip silently if not available)
3. Record start timestamp
4. Launch `serve.sh` in background
5. Poll `GET http://localhost:$VLLM_PORT/v1/models` every 1 second until HTTP 200
6. Record ready timestamp
7. Output: `{start_epoch, ready_epoch, duration_s}`

Save to `results/<env>/<ts>/cold_start/run_<n>.json`. Warmup applies here too (first run discarded).

After all cold-start runs complete, restart serve.sh one final time and leave it running (subsequent benchmarks may depend on it, or the operator may want to inspect).

### Tier 3: Storage Throughput (bench_fio.sh)

Model loading and checkpointing performance. Four profiles, all with `direct=1` (bypass OS page cache), `ioengine=libaio`:

| Profile | Pattern | Block size | Jobs | iodepth | Simulates |
|---|---|---|---|---|---|
| seq_read | read | 1M | 1 | 32 | Model loading from disk |
| seq_write | write | 1M | 1 | 32 | Checkpoint saving |
| rand_read | randread | 256K | 4 | 16 | Dataset loading |
| rand_write | randwrite | 4K | 4 | 32 | Docker layer extraction |

Each profile runs for FIO_RUNTIME seconds against FIO_SIZE test files in FIO_DIR. Output: fio's native JSON format saved to `results/<env>/<ts>/fio/<profile>.json`.

fio runs once per profile (no repetitions needed -- fio's internal time_based averaging is sufficient for steady-state IO).

### Tier 4: CPU + Cgroup Throttle (bench_cpu.sh)

Detect silent container CPU throttling that RunPod's cgroup policies might impose.

Two measurements:
1. **sysbench cpu**: Run `sysbench cpu --threads=<auto> --time=<duration> run`. Auto-detect thread count from `nproc`. Capture events/s and avg/p95 latency. Save to `results/<env>/<ts>/cpu/sysbench.json`.

2. **cgroup throttle delta**: Read `/sys/fs/cgroup/cpu.stat` (cgroup v2, preferred) or `/sys/fs/cgroup/cpu/cpu.stat` (cgroup v1, fallback) before and after the vLLM serving benchmark. Compute deltas for `nr_throttled` and `throttled_usec`. If neither file exists (not in a cgroup), record nulls and log a warning. Save to `results/<env>/<ts>/cpu/cgroup_throttle.json`.

The cgroup snapshot is taken by run_all.sh, which reads cpu.stat before bench_vllm_serving.sh and after it completes, then writes the delta.

## Environment Manifest (capture_manifest.sh)

Captured once per run into `results/<env>/<ts>/manifest.json`. Fields:

| Field | Source |
|---|---|
| hostname | `hostname` |
| in_container | Presence of `/.dockerenv` or `/run/.containerenv` |
| kernel | `uname -r` |
| distro | `/etc/os-release` |
| gpu_model | `nvidia-smi --query-gpu=name --format=csv,noheader` (first GPU) |
| gpu_count | `nvidia-smi --query-gpu=name --format=csv,noheader | wc -l` |
| gpu_driver | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` (first) |
| cuda_version | `nvidia-smi` parsed from header |
| cpu_model | `/proc/cpuinfo` model name |
| cpu_count | `nproc` |
| memory_gb | `free -g` total |
| vllm_version | `python -c "import vllm; print(vllm.__version__)"` |
| torch_version | `python -c "import torch; print(torch.__version__)"` |
| python_version | `python --version` |
| cgroup_cpu_quota | `/sys/fs/cgroup/cpu.max` (cgroup v2) or `/sys/fs/cgroup/cpu/cpu.cfs_quota_us` (v1) |
| cgroup_memory_limit | `/sys/fs/cgroup/memory.max` (v2) or `/sys/fs/cgroup/memory/memory.limit_in_bytes` (v1) |
| config_snapshot | Full dump of all config.sh values |
| timestamp | ISO 8601 UTC |

## Output Structure

```
results/
  <env-name>/
    <ISO-timestamp>/
      manifest.json
      vllm_serving/
        rate_1/
          run_0.json    # warmup (discarded by summarize)
          run_1.json
          ...
          run_5.json
        rate_2/
          ...
        rate_inf/
          ...
      cold_start/
        run_0.json      # warmup
        run_1.json
        ...
      fio/
        seq_read.json
        seq_write.json
        rand_read.json
        rand_write.json
      cpu/
        sysbench.json
        cgroup_throttle.json
      summary.json
      summary.txt
```

## summarize.py

Reads all raw JSON from a run directory and aggregates into `summary.json` + `summary.txt`.

### vLLM Serving

Per request rate, drops warmup runs, computes across measurement runs:
- **Median and p95** of: request throughput (req/s), token throughput (tok/s), TTFT p50, TTFT p95, TTFT p99, TPOT p50, TPOT p95, ITL p50, ITL p95

### Cold Start

Drops warmup, computes median and p95 of `duration_s`.

### FIO

Per profile: bandwidth (MB/s), IOPS, latency p50/p99. Single run per profile so no aggregation needed -- just extract and reformat.

### CPU

sysbench: events/s, latency avg and p95. cgroup: raw deltas for nr_throttled and throttled_usec.

### Output format

`summary.json`: structured dict keyed by benchmark tier, then by metric. Machine-readable for compare.py.

`summary.txt`: Human-readable tables. One section per tier. Example:

```
=== vLLM Serving (median across 5 runs) ===
Rate  | Tput req/s | Tput tok/s | TTFT p50 | TTFT p95 | TPOT p50
------+------------+------------+----------+----------+---------
1     |  1.0       |  128.0     |  42ms    |  48ms    |  8.2ms
2     |  2.0       |  256.0     |  43ms    |  51ms    |  8.3ms
...

=== Cold Start (5 runs) ===
Median: 28.3s | p95: 30.1s

=== Storage (fio) ===
Profile    | BW (MB/s) | IOPS    | Lat p50  | Lat p99
-----------+-----------+---------+----------+--------
seq_read   | 3200      | 3200    | 0.3ms    | 1.2ms
...

=== CPU ===
sysbench: 12400 events/s, avg lat 0.08ms, p95 lat 0.10ms
cgroup throttle: nr_throttled=0, throttled_usec=0
```

## compare.py

Takes two run directories as arguments. Loads `summary.json` and `manifest.json` from each.

### Pre-flight checks
- Warn (do not fail) if GPU model, GPU count, or vLLM version differ between runs
- Print both manifests side-by-side as a header

### Output table

One row per request rate per metric, showing baseline value, treatment value, and percent delta. Color-coded in terminal: green if treatment is within 5% of baseline, yellow if 5-15%, red if >15% degradation.

Separate sections for cold start, fio, and CPU with the same delta format.

```
=== Manifest Comparison ===
                  | ec2-direct          | runpod-pod
------------------+---------------------+--------------------
hostname          | ip-10-0-1-42        | 6f3a2b1c8d9e
gpu_model         | NVIDIA L4           | NVIDIA L4
gpu_count         | 8                   | 8
vllm_version      | 0.8.3               | 0.8.3
cgroup_cpu_quota  | max (unlimited)     | 800000 (8 cores)
...

=== vLLM Serving ===
Rate | Metric     | ec2-direct | runpod-pod | Delta
-----+------------+------------+------------+--------
1    | tput req/s |  1.0       |  1.0       |  +0.0%
1    | tput tok/s |  128.0     |  127.5     |  -0.4%
1    | TTFT p50   |  42ms      |  43ms      |  +2.4%
...

=== Cold Start ===
Metric | ec2-direct | runpod-pod | Delta
-------+------------+------------+-------
median |  28.3s     |  29.1s     | +2.8%
p95    |  30.1s     |  31.5s     | +4.6%

=== Storage (fio) ===
Profile   | Metric   | ec2-direct | runpod-pod | Delta
----------+----------+------------+------------+-------
seq_read  | BW MB/s  | 3200       | 3180       | -0.6%
...

=== CPU ===
Metric          | ec2-direct | runpod-pod | Delta
----------------+------------+------------+-------
events/s        | 12400      | 12350      | -0.4%
throttled_usec  | 0          | 1200       | +1200 (Treatment only)
```

## File Layout

```
runpod-bench/
  README.md              # Usage instructions
  config.sh              # All parameters (env vars with defaults)
  launch_baseline.sh     # Host-side: docker run with correct flags for baseline (A)
  setup.sh               # One-time: install fio/sysbench/vllm, pre-warm HF cache
  serve.sh               # Launch vllm serve with consistent flags
  capture_manifest.sh    # Emit manifest.json
  bench_vllm_serving.sh  # Sweep request rates x runs
  bench_cold_start.sh    # Stop/start server, time readiness
  bench_fio.sh           # 4 fio profiles
  bench_cpu.sh           # sysbench + cgroup throttle delta
  run_all.sh             # Top-level orchestrator
  summarize.py           # Aggregate raw JSON -> summary.{json,txt}
  compare.py             # Diff two summaries -> delta table
  results/               # Created at runtime (gitignored)
```

## Script Dependencies

```
setup.sh installs:
  - fio (apt)
  - sysbench (apt)
  - libaio-dev (apt, required for fio's libaio ioengine inside containers)
  - jq (apt, JSON parsing in bash scripts)
  - vllm (pip)
  - huggingface_hub (pip, for cache warmup)

Pre-installed in the pytorch image:
  - Python 3.11, pip, torch, CUDA toolkit
  - nvidia-smi

setup.sh also:
  - Validates HF_TOKEN is set
  - Validates nvidia-smi works
  - Records installed vllm version to config snapshot
  - Pre-warms HF cache: downloads model weights so benchmarks don't measure download time
```

## Error Handling

Every script follows these conventions:
- `set -euo pipefail` at the top
- Fail fast with a clear message if: HF_TOKEN unset, nvidia-smi fails, vLLM server unreachable when expected, disk full
- Non-critical failures (page cache drop needs root, cgroup files missing) log a warning and continue with nulls
- All stdout/stderr from benchmark tools is tee'd to a log file alongside the JSON output

## Non-Goals

- Multi-node / multi-GPU / NCCL benchmarks
- Training workloads
- Network benchmarks across regions (loopback only)
- Comparing different GPU types
- Pod scheduling latency (separate from this harness -- measure manually if needed)
- Automated A/B interleaving (README recommends it, harness doesn't enforce it)
