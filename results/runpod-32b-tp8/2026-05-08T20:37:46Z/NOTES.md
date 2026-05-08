# Runpod-pod, 8× L4, Qwen3-32B, TP=8 — vLLM serving sweep, 2026-05-08

## Status: paired with EC2-direct 2026-05-07 baseline

This is the Runpod 8-GPU pod side of the 32B/TP=8 comparison. The EC2-direct
baseline lives at `results/ec2-direct-32b-tp8/2026-05-07T17:48:31Z/`. Both
runs target the same physical machine (i-017fc2665bb14dd86, 8× L4) using the
same Docker image, the same vLLM version, and an identical workload — the only
variable is the Runpod orchestration layer.

A `COMPARISON.md` lives in the same directory as this NOTES; it has the
rate-by-rate delta table for anyone who wants the raw numbers.

## What this test answers

**Primary question:** When a customer brings their own EC2 instance to Runpod
and runs a realistic 8-GPU TP=8 inference workload through a Runpod-managed
pod, do they pay a meaningful performance tax on throughput, latency, or
container-layer CPU?

This is the headline answer for prospective Runpod Anywhere (Runpod
Everywhere / RPE) customers: *"if I bring my g6.48xlarge to Runpod, what
does Runpod's orchestration layer cost me?"*

## What this test does NOT answer

- Whether the same finding holds at 70B+ scale on different hardware.
- Cold start, storage I/O, network — those tiers were skipped to focus on
  steady-state inference.
- Long-context behavior (we tested 512 input / 128 output).
- Multi-tenant noisy-neighbor scenarios (we ran on a private pool, no
  contention).
- DP=8 vs TP=8 topology comparisons.
- Whether the benign cgroup result holds under sustained 24+ hour load.

## What we ran

| | |
|--|--|
| Model | Qwen/Qwen3-32B (TP=8 sharded across 8× L4) |
| Backend | vLLM 0.16.0 (pinned), torch 2.9.1+cu128 |
| Image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| Pod | `ql5gaiccidrhl3` on `audry-i-017fc2665bb14dd86-g6.48xlarge-3`, secure private pool |
| GPUs | 8× NVIDIA L4, driver 580.126.20 |
| cgroup | `cpu.max = 16320000 100000` (≈163 vCPU cap), `memory.max = 728 GiB` |
| Workload | INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=200, seed=42 |
| Reps | 1 warmup + 5 measured runs per rate |
| Rates | 16, 32, 64, 128, inf |
| `enforce_eager` | true |

The cgroup CPU cap is the only meaningful environment difference vs the
EC2-direct baseline (which had `cpu.max = max`). Memory was effectively
matched (728 GiB cap on Runpod side, host's full 728 GiB on EC2 side).

## Headline numbers (median across 5 measured runs)

| Rate | Tput req/s | Tput tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) |
|------|-----------:|-----------:|--------------:|--------------:|--------------:|
| 16   |        5.1 |      651.6 |         499.1 |       1,001.9 |         228.3 |
| 32   |        5.4 |      697.1 |         602.3 |       1,179.4 |         247.8 |
| 64   |        5.6 |      720.2 |         845.4 |       1,540.5 |         257.5 |
| 128  |        5.7 |      727.8 |       1,206.7 |       2,168.9 |         260.4 |
| inf  |        5.7 |      728.0 |       2,915.9 |       3,270.8 |         253.5 |

cgroup throttle during the sweep: `nr_throttled=0`, `throttled_usec=0`.

## Headline finding: the orchestration layer is free

**At every rate, on every metric, Runpod-pod is within ±2% of the
EC2-direct baseline.** The single exception is TTFT p99 at rate=`inf`
(burst), which is +6% — and even that is bounded by deep-queue tail
behavior, not orchestration overhead. Throughput, steady-state latency,
and median TTFT are statistically identical between the two
environments.

| Metric (median across rates 16/32/64/128/inf) | EC2-direct | Runpod-pod | Δ |
|---|---:|---:|---:|
| Saturation throughput (tok/s, max across rates) | 731.1 | 728.0 | **−0.4%** |
| Steady-state TPOT p50 (ms, mean across rates) | 248.4 | 249.5 | **+0.4%** |
| TTFT p50 at rate=64 (ms) | 830.1 | 845.4 | **+1.8%** |
| cgroup `throttled_usec` over the entire sweep | 0 | 0 | **0** |

For the full per-rate / per-metric comparison see `COMPARISON.md`.

## Observations from this run alone

1. **Saturation throughput: ~728 tok/s.** Reached around rate=32, flat
   through 64, 128, and `inf`. Identical saturation point to the
   EC2-direct baseline, identical absolute number within noise.

2. **Per-token decode latency: ~250 ms.** Dominated by GPU compute and
   NCCL all-reduce on every layer (TP=8). Container layer adds nothing
   measurable here — the GPUs are doing the same work in the same time
   regardless of orchestration.

3. **TTFT scales gracefully through rate=128 and degrades only at burst
   (rate=`inf`).** Same shape as EC2-direct. Both environments have the
   same queue dynamics; bombardment is bombardment.

4. **Zero CPU cgroup throttling, despite a 163-vCPU cap.** Runpod's
   private-pool pod was given enough room (163 of 192 host vCPUs, 85%)
   that the workload never came close to hitting it. The workload itself
   is GPU-bound — sustained CPU usage in vLLM serving for 32B/TP=8 sits
   well below the cap.

5. **The +6% TTFT p99 at rate=`inf` is not orchestration overhead.**
   That's tail-behavior of a deep queue under bombardment. Looking at
   the per-run values (3304 / 3069 / 3271 / 3243 / 3329 ms), the spread
   is large. Five samples isn't enough to call a 6% delta a real signal
   — confidence intervals at this rate easily span ±5%.

## How this compares against the EC2-direct baseline

The EC2-direct NOTES laid out the prediction: *"If the Runpod pod
matches these absolute numbers, the container layer is neutral for
realistic 8-GPU TP=8 workloads."* This is exactly what we measured.

| Prediction | Actual | Verdict |
|---|---|---|
| Saturation throughput within ±5% | −0.4% | ✓ Confirmed |
| Steady-state TPOT within ±5% | +0.4% | ✓ Confirmed |
| TTFT p50 within ±5% across all rates | +1.8% worst case | ✓ Confirmed |
| cgroup throttle = 0 | 0 | ✓ Confirmed |

## What this means for the RPE pitch

Cleared messages we can use with prospective customers:

> **"On 8× L4 with a Qwen3-32B inference workload, running through
> Runpod's pod orchestration costs less than 1% throughput and less
> than 2% on median latency vs. running Docker directly on the same
> EC2 instance. CPU throttling never fires. The container layer is
> functionally free."**

The numbers also reinforce what the 1-GPU 8B paired comparison already
suggested: the cost of "bring your EC2 to Runpod" is dominated by what
the customer was already going to pay (Docker, NCCL, vLLM) — not by
anything Runpod's layer adds on top.

## Caveats to keep in mind

- This is one model architecture (Qwen) on one GPU type (L4). Larger
  models on H100/H200 may behave differently, especially if they're
  bandwidth-bound rather than compute-bound.
- Single tenant on a private pool. A noisy-neighbor scenario in shared
  infra could change the throttle picture. The harness is designed to
  detect that — re-run on a different pool config to test.
- 30-minute total benchmark window; longer sustained load may surface
  effects that don't show up in 5-minute rate runs (memory leaks,
  thermal throttling, etc.). Worth a 24h soak for due diligence before
  a high-stakes RPE customer commits.
- The EC2-direct baseline was run on 2026-05-07; the Runpod-pod run was
  2026-05-08. ~24h apart. Same physical machine, same image, same
  configuration — but technically there's a small "different day"
  factor. Inference workloads are deterministic enough that this
  doesn't seem to have mattered (numbers are within noise), but worth
  noting.

## How to reproduce

```bash
# On a fresh Runpod pod with 8× L4
git clone https://github.com/audryhsu/runpod-bench.git && cd runpod-bench
export HF_TOKEN=<your-hf-token>
export MODEL=Qwen/Qwen3-32B
VLLM_VERSION=0.16.0 ./setup.sh
VLLM_TP_SIZE=8 ./serve.sh > /tmp/vllm.log 2>&1 &
while ! curl -sf http://localhost:8000/v1/models > /dev/null; do sleep 10; done
BENCH_RUNS=5 NUM_PROMPTS=200 REQUEST_RATES="16 32 64 128 inf" VLLM_TP_SIZE=8 \
  ./run_all.sh --env-name runpod-32b-tp8 --skip-fio --skip-cpu --skip-cold-start
```

## Files

- [`manifest.json`](./manifest.json) — host/container/version snapshot
- [`summary.json`](./summary.json) / [`summary.txt`](./summary.txt) — per-rate aggregate metrics
- [`vllm_serving/`](./vllm_serving/) — per-run vLLM bench-serve output
- [`cpu/cgroup_throttle.json`](./cpu/cgroup_throttle.json) — pre/post throttle counters (zero deltas)
- [`COMPARISON.md`](./COMPARISON.md) — full rate-by-rate delta table vs EC2-direct
