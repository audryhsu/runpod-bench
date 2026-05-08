# EC2-direct, 8× L4, Qwen3-32B, TP=8 — vLLM serving sweep, 2026-05-07

## Status: EC2-direct baseline only — Runpod 8-GPU pod side TBD

This run captures the EC2-direct numbers for a realistic 8-GPU TP=8
deployment of Qwen3-32B (a model that genuinely fills 8 L4s — unlike
the prior 8B/TP=8 attempt that was scrapped). The paired Runpod 8-GPU
pod run has not been executed yet; once it is, drop the results next to
this run and produce a `COMPARISON.md`.

## What this test answers

**Primary question:** What is the EC2-direct (uncapped CPU) baseline
performance for a realistic 8-GPU TP=8 inference workload? What's the
saturation throughput, the steady-state TTFT/TPOT profile, and the
burst behavior?

This is the upper-bound reference. It's what an unconstrained 8-L4
host can do with Qwen3-32B before any container-layer overhead is
added. The Runpod-pod-side run gets compared against this.

## What this test does NOT answer

- Anything about Runpod 8-GPU pod performance. (EC2-direct half only.)
- Whether DP=8 of a smaller model would scale better. (Different
  topology question.)
- Whether 70B-class models would behave differently.
- Anything about non-Qwen architectures.
- Cold start, storage I/O, or network. (Skipped tiers.)
- Whether longer contexts (e.g., 32k input) change the picture.
- Whether the 1-GPU "container layer is free" finding generalizes to
  multi-GPU. (Will get suggested by the eventual paired comparison;
  not provable from a single EC2-direct run alone.)

## What we ran

| | |
|--|--|
| Model | Qwen/Qwen3-32B (~32B params, ~64 GB bf16 weights → 7.69 GB/GPU at TP=8) |
| Backend | vLLM 0.16.0 (pinned), torch 2.9.1+cu128 |
| Topology | TP=8 (single vLLM instance, model sharded across all 8 L4s) |
| GPUs | 8× NVIDIA L4 on a single EC2 host, driver 580.126.20 |
| Container | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, `--userns=host`, `--gpus all` |
| cgroup | `cpu.max = max`, `memory.max = max` (no caps) |
| Workload | INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=200, seed=42 |
| Reps | 1 warmup + 5 measured runs per rate |
| Rates | 16, 32, 64, 128, inf |
| `enforce_eager` | true (CUDA graphs disabled) |
| Max KV-cache concurrency reported by vLLM | 47.58× at 8128 tokens per request |

## Headline numbers (median across 5 measured runs)

| Rate | Tput req/s | Tput tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) |
|------|-----------:|-----------:|--------------:|--------------:|--------------:|
| 16   |        5.1 |      653.5 |         500.3 |         991.5 |         226.9 |
| 32   |        5.5 |      699.5 |         610.3 |       1,161.4 |         246.8 |
| 64   |        5.6 |      722.9 |         830.1 |       1,562.2 |         256.7 |
| 128  |        5.7 |      730.9 |       1,192.5 |       2,112.4 |         259.5 |
| inf  |        5.7 |      731.1 |       3,060.4 |       3,084.5 |         251.4 |

cgroup throttle during the sweep: `nr_throttled=0`, `throttled_usec=0`.

## Observations from this run alone

1. **Saturation throughput: ~731 tok/s.** Reached around rate=32; rates
   64, 128, inf are indistinguishable on throughput. The 8-L4 / TP=8 /
   32B configuration is offered-rate-saturated by 32 req/s for this
   workload.
2. **Per-token decode latency: ~250 ms.** That's 2.5× the 1-GPU 8B
   number (97 ms). Two contributing factors: (a) the model is 4× larger,
   (b) TP=8 adds NCCL all-reduce on every layer.
3. **TTFT scales gracefully through rate=128** (under 1.2 s p50, under
   2.2 s p99) and spikes hard at rate=`inf` (3.06 s p50, 3.08 s p99).
   Bombardment is the worst-case scenario for queueing — 200 prompts
   landing simultaneously have to wait through a deep queue before
   their prefill gets scheduled.
4. **TPOT is essentially flat from rate=16 onward** (227 → 251 ms).
   Once decode is in steady state, the per-token cost is GPU-bound and
   doesn't change with offered load.
5. **No cgroup throttling on EC2-direct.** Expected — `cpu.max = max`.
   The interesting question is whether a Runpod 8-GPU pod stays
   throttle-free under the same workload.

## How this compares against expectation

| Prediction | Actual | Verdict |
|------------|--------|---------|
| Saturation throughput 1,500–3,000 tok/s | 731 tok/s | **Lower than predicted.** 32B per-token compute is heavier than I'd assumed; TP=8 efficiency is also still bounded by NCCL even with full GPU utilization. |
| TPOT 120–160 ms | 251 ms | **Higher.** Same reason — larger model means more per-token GPU work, and NCCL adds proportionally. |
| Graceful TTFT through rate=64, burst spike at inf | confirmed | ✓ |
| cgroup throttle = 0 | 0 | ✓ |

The "lower than predicted" results don't invalidate the test — they
just calibrate expectations for the Runpod-side comparison. If the
Runpod pod matches these absolute numbers, the container layer is
neutral for realistic 8-GPU TP=8 workloads.

## How to add the Runpod-pod side

Provision an 8-GPU Runpod pod, ssh in, and run:

```bash
export HF_TOKEN=<token>
export MODEL=Qwen/Qwen3-32B
git clone <this repo> && cd runpod-bench
VLLM_VERSION=0.16.0 ./setup.sh   # downloads 32B model (~64 GB)
VLLM_TP_SIZE=8 ./serve.sh > /tmp/vllm.log 2>&1 &
# wait until http://localhost:8000/v1/models responds
BENCH_RUNS=5 NUM_PROMPTS=200 REQUEST_RATES="16 32 64 128 inf" VLLM_TP_SIZE=8 \
  ./run_all.sh --env-name runpod-32b-tp8 --skip-fio --skip-cpu --skip-cold-start
```

Then drop the resulting `runpod-32b-tp8/<timestamp>/` directory next to
this EC2-direct run, and write a `COMPARISON.md` similar to the 1-GPU
run's. The harness's manifest will capture the matching env (vLLM/torch
versions, cgroup limits, GPU count) so the comparison is fair.

## Files

- [`ec2-direct/2026-05-07T17:48:31Z/manifest.json`](./ec2-direct/2026-05-07T17:48:31Z/manifest.json) — full host/container/version snapshot
- [`ec2-direct/2026-05-07T17:48:31Z/summary.json`](./ec2-direct/2026-05-07T17:48:31Z/summary.json) / [`ec2-direct/2026-05-07T17:48:31Z/summary.txt`](./ec2-direct/2026-05-07T17:48:31Z/summary.txt) — per-rate aggregate metrics
- [`ec2-direct/2026-05-07T17:48:31Z/vllm_serving/`](./ec2-direct/2026-05-07T17:48:31Z/vllm_serving/) — per-run vLLM bench-serve output, organized by rate
- [`ec2-direct/2026-05-07T17:48:31Z/cpu/cgroup_throttle.json`](./ec2-direct/2026-05-07T17:48:31Z/cpu/cgroup_throttle.json) — pre/post throttle counters (zero)
