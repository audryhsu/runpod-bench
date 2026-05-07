# RunPod Docker vLLM benchmark — 2026-05-07

## Environment
- **Host**: RunPod pod, container hostname `c849e5c7b38a`
- **GPU**: 1× NVIDIA L4 (driver 580.126.20, compute cap 8.9)
- **CPU**: AMD EPYC 7R13, 192 vCPUs visible
- **Memory**: 728 GB visible
- **cgroup CPU quota**: 2040000 / 100000 = 20.4 effective cores
- **cgroup memory limit**: ~83.8 GiB
- **Distro / Kernel**: Ubuntu 22.04.5 / 6.17.0-1012-aws
- **vLLM**: 0.16.0 (torch 2.9.1+cu128, Python 3.11.11)
- **Model**: `/workspace/qwen3-8b` (Qwen3-8B local weights, served by Docker vLLM at localhost:8000)

## How it was run

vLLM was already running in a separate Docker container on the same host, listening on `localhost:8000` and serving the local-path model id `/workspace/qwen3-8b`. The harness was driven from a sibling container on the same pod (so `localhost:8000` was reachable directly).

Exact invocation:

```bash
MODEL=/workspace/qwen3-8b \
HF_TOKEN=dummy \
BENCH_RUNS=2 \
NUM_PROMPTS=50 \
./run_all.sh --env-name runpod-docker-vllm --skip-fio --skip-cpu --skip-cold-start
```

- `MODEL=/workspace/qwen3-8b`: must match the model id the Docker vLLM is serving (visible via `curl localhost:8000/v1/models`); the harness's default `Qwen/Qwen3-8B` would be rejected.
- `HF_TOKEN=dummy`: `config.sh` requires this to be set, but the model is local so the value is unused.
- `--skip-cold-start`: the cold-start tier `pkill -f "vllm serve"` and re-launches via `serve.sh`, which would fight the Docker-managed vLLM. Skipped.
- `--skip-fio --skip-cpu`: storage and CPU tiers are unrelated to the vLLM workload — skipped to keep this run scoped.

## Config (shortened from defaults to keep wall time ~6 min)
- `INPUT_LEN=512`, `OUTPUT_LEN=128` (defaults)
- `NUM_PROMPTS=50` (default 200) — overridden
- `BENCH_RUNS=2` (default 5) — overridden
- `WARMUP_RUNS=1` (default)
- `REQUEST_RATES="1 2 4 8 16 inf"` (default)
- `SEED=42`, `enforce_eager=1`, `gpu_mem_util=0.95`, `max_model_len=8128`
- Total runs: 6 rates × 3 runs (1 warmup + 2 measured) = 18 runs

## Patches applied to the harness before this run
- `capture_manifest.sh`: replaced `grep "model name" /proc/cpuinfo | head -1 | cut ...` with a single `awk` to avoid SIGPIPE-under-`pipefail` killing the manifest step. This patch is uncommitted on the harness branch as of this run.
- Installed `jq` via apt (was missing on this pod); the harness expects it.

## Headline results (median across measured runs)

| Rate    | Tput req/s | Tput tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) |
|---------|-----------:|-----------:|--------------:|--------------:|--------------:|
| 1       |        0.8 |      108.7 |         302.1 |         479.5 |          76.0 |
| 2       |        1.5 |      186.6 |         238.6 |         583.3 |          82.4 |
| 4       |        2.2 |      278.0 |         236.8 |         667.2 |          88.7 |
| 8       |        2.8 |      358.9 |         231.7 |         789.6 |          93.8 |
| 16      |        2.9 |      375.2 |         271.5 |         840.5 |          97.5 |
| inf     |        3.2 |      406.6 |        1274.6 |        1341.3 |          97.3 |

## Observations
- Throughput saturates between rate=8 and rate=16 (~3 req/s, ~400 tok/s).
- TTFT p99 climbs steadily from rate=1 onwards and jumps sharply at rate=inf as requests queue.
- **cgroup throttle** during the sweep: `nr_throttled=61`, `throttled_usec=360,120,699` (~360 ms total). Non-zero, suggesting the pod's CPU quota was hit briefly. Worth comparing against bare-EC2 numbers tomorrow to see if this is the source of any RunPod-specific overhead.

## Caveats
- Only 2 measured runs per rate — numbers are noisier than a full default run would produce. For publishable comparisons re-run with `BENCH_RUNS=5 NUM_PROMPTS=200` (~40 min wall time).
- `enforce_eager=1` disables CUDA graphs; latency would improve modestly with it off.
