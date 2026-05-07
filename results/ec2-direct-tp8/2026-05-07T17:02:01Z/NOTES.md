# EC2-direct, 8× L4, TP=8 — vLLM serving sweep, 2026-05-07

## Status: EC2-direct baseline only — Runpod 8-GPU pod side TBD

This run captures the EC2-direct numbers for an 8-GPU TP=8 vLLM
deployment of Qwen3-8B. The paired Runpod 8-GPU pod run has not been
executed yet; once it is, drop the results next to this run and produce
a `COMPARISON.md` similar to the 1-GPU run.

## What we ran

| | |
|--|--|
| Model | Qwen/Qwen3-8B |
| Backend | vLLM 0.16.0 (pinned), torch 2.9.1+cu128 |
| Topology | TP=8 (single vLLM instance, 1 model sharded across 8 GPUs) |
| GPUs | 8× NVIDIA L4 on a single EC2 host (driver 580.126.20) |
| Container | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, `--userns=host`, `--gpus all` |
| cgroup | `cpu.max = max`, `memory.max = max` (no caps) |
| Workload | INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=200, seed=42 |
| Reps | 1 warmup + **5 measured runs per rate** |
| Rates | **16, 32, 64, 128, inf** (skipped low rates from the 1-GPU run since they're guaranteed GPU-bound and uninformative) |
| `enforce_eager` | true (CUDA graphs disabled) |

## Headline numbers (median across 5 measured runs)

| Rate | Tput req/s | Tput tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) |
|------|-----------:|-----------:|--------------:|--------------:|--------------:|
| 16   |        9.0 |    1,158.1 |         210.2 |         366.9 |          91.5 |
| 32   |        9.8 |    1,260.2 |         230.8 |         490.3 |         118.6 |
| 64   |       10.3 |    1,322.5 |         272.9 |         552.2 |         131.3 |
| 128  |       10.5 |    1,350.1 |         348.3 |         704.7 |         138.0 |
| inf  |       10.6 |    1,350.8 |       1,218.8 |       1,561.5 |         139.3 |

cgroup throttle during the sweep: `nr_throttled=0`, `throttled_usec=0`.

## Observations from this run alone (no comparison yet)

1. **Saturation throughput ≈ 1,350 tok/s, reached around rate=64.** Rate
   128 and rate=`inf` are indistinguishable on throughput — the system
   is already at its ceiling at rate=64.
2. **8 GPUs deliver ~3.3× the 1-GPU peak (412 → 1,350 tok/s), not 8×.**
   Two reasons:
   - **NCCL overhead**: TP all-reduce on every layer adds CPU-side
     coordination work and inter-GPU synchronization that scales
     superlinearly with GPU count.
   - **Per-GPU underutilization**: an 8B model TP-sharded 8 ways gives
     each GPU only ~1B of compute per step. Most of each L4 sits idle
     waiting for NCCL.
   For "is the entire 8-GPU node being exercised?" — yes, in the sense
   that all 8 are active, but the workload isn't large enough to
   saturate them on compute. A 32B+ model would utilize them better.
3. **TPOT increased from 97 ms (TP=1) to 139 ms (TP=8) — +43 %.** Per-
   token decode latency is now bottlenecked by the per-layer all-reduce.
   This is intrinsic to TP, not container-related.
4. **TTFT scales gracefully** under steady load (210–700 ms across rates
   16–128) but spikes to 1.56 s under bombardment (rate=`inf`). The
   bombardment number is *higher* than the 1-GPU bombardment (1.08 s),
   despite the higher peak throughput — at saturation there are simply
   more requests in flight competing to be admitted.
5. **No cgroup throttling on EC2-direct** — expected, since `cpu.max =
   max`. The interesting question is whether a Runpod 8-GPU pod stays
   throttle-free under TP=8's NCCL coordination load. We don't have
   that data yet.

## What this run does and does not tell us

**Does tell us:**
- The absolute capabilities of an 8× L4 host serving Qwen3-8B at TP=8,
  with no container-layer constraint. This is the upper-bound reference
  point.
- TP=8 of an 8B model has poor scaling efficiency (~3.3×, not 8×) on
  this hardware. If a customer's goal is throughput, this is the wrong
  topology — DP=8 (8 single-GPU replicas) would deliver closer to 8×.

**Does not tell us:**
- Anything about Runpod 8-GPU pod performance. The whole reason to run
  this was to set up the EC2-direct baseline; the comparison requires
  the matching Runpod-pod run.
- Whether a bigger model (32B, 70B) would change the conclusion. With a
  bigger model each GPU has more useful work to do, NCCL overhead
  becomes a smaller fraction, and CPU pressure also changes.

## How to add the Runpod-pod side

When an 8-GPU Runpod pod is provisioned, ssh in and run:

```bash
export HF_TOKEN=<token>
git clone <this repo> && cd runpod-bench
VLLM_VERSION=0.16.0 ./setup.sh
VLLM_TP_SIZE=8 ./serve.sh > /tmp/vllm.log 2>&1 &
# wait until http://localhost:8000/v1/models responds
BENCH_RUNS=5 NUM_PROMPTS=200 REQUEST_RATES="16 32 64 128 inf" VLLM_TP_SIZE=8 \
  ./run_all.sh --env-name runpod-tp8 --skip-fio --skip-cpu --skip-cold-start
```

Then drop the resulting `runpod-tp8/<timestamp>/` directory next to this
EC2-direct run and write a `COMPARISON.md`. The harness will capture the
matching manifest (vLLM/torch versions, cgroup limits, GPU count) so the
comparison is fair.

## Files

- [`manifest.json`](./manifest.json) — full host/container/version snapshot
- [`summary.json`](./summary.json) / [`summary.txt`](./summary.txt) — per-rate aggregate metrics
- [`vllm_serving/`](./vllm_serving/) — per-run vLLM bench-serve output, organized by rate
- [`cpu/cgroup_throttle.json`](./cpu/cgroup_throttle.json) — pre/post throttle counters (zero)
