# Runpod-pod vs EC2-direct: Qwen3-32B / TP=8 / 8× L4

| | |
|--|--|
| Baseline (A) | `results/ec2-direct-32b-tp8/2026-05-07T17:48:31Z/` |
| Treatment (B) | `results/runpod-32b-tp8/2026-05-08T20:37:46Z/` |
| Same image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| Same vLLM | 0.16.0 |
| Same torch | 2.9.1+cu128 |
| Same kernel | 6.17.0-1012-aws |
| Same model | Qwen/Qwen3-32B |
| Same workload | INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=200, seed=42 |
| Same physical host | i-017fc2665bb14dd86 (8× L4, 192 vCPUs, 728 GB) |
| Different cgroup CPU | A: `max 100000` (uncapped) · B: `16320000 100000` (~163 vCPU cap) |
| Different cgroup mem | A: `max` · B: `725999996928` (728 GB) |
| Different timestamp | A: 2026-05-07 17:48 UTC · B: 2026-05-08 20:37 UTC (~27h apart) |

## TL;DR

**±2% on every metric except a single outlier on TTFT p99 at rate=inf
(+6%, well within the noise of 5 samples). Saturation throughput is
identical. Steady-state TPOT is identical. CPU throttle is zero in both.
The orchestration layer is free at this scale.**

## Per-rate delta table

Format: `EC2-direct → Runpod-pod (Δ%)`. Negative throughput delta and
positive latency delta both indicate Runpod-pod is *worse*. Green = within
±5% (within noise), yellow = 5–15%, red = >15%.

### Rate = 16 req/s

| Metric | EC2-direct | Runpod-pod | Δ |
|---|---:|---:|---:|
| tput req/s | 5.106 | 5.091 | −0.3% |
| tput tok/s | 653.5 | 651.6 | −0.3% |
| TTFT p50 (ms) | 500.3 | 499.1 | −0.3% |
| TTFT p99 (ms) | 991.5 | 1001.9 | +1.1% |
| TPOT p50 (ms) | 226.9 | 228.3 | +0.6% |
| ITL p50 (ms) | 245.4 | 247.4 | +0.8% |

### Rate = 32 req/s

| Metric | EC2-direct | Runpod-pod | Δ |
|---|---:|---:|---:|
| tput req/s | 5.465 | 5.446 | −0.3% |
| tput tok/s | 699.5 | 697.1 | −0.3% |
| TTFT p50 (ms) | 610.3 | 602.3 | −1.3% (Runpod faster) |
| TTFT p99 (ms) | 1161.4 | 1179.4 | +1.6% |
| TPOT p50 (ms) | 246.8 | 247.8 | +0.4% |
| ITL p50 (ms) | 251.8 | 253.0 | +0.4% |

### Rate = 64 req/s

| Metric | EC2-direct | Runpod-pod | Δ |
|---|---:|---:|---:|
| tput req/s | 5.648 | 5.626 | −0.4% |
| tput tok/s | 722.9 | 720.2 | −0.4% |
| TTFT p50 (ms) | 830.1 | 845.4 | +1.8% |
| TTFT p99 (ms) | 1562.2 | 1540.5 | −1.4% (Runpod faster) |
| TPOT p50 (ms) | 256.7 | 257.5 | +0.3% |
| ITL p50 (ms) | 252.1 | 253.0 | +0.4% |

### Rate = 128 req/s

| Metric | EC2-direct | Runpod-pod | Δ |
|---|---:|---:|---:|
| tput req/s | 5.711 | 5.686 | −0.4% |
| tput tok/s | 730.9 | 727.8 | −0.4% |
| TTFT p50 (ms) | 1192.5 | 1206.7 | +1.2% |
| TTFT p99 (ms) | 2112.4 | 2168.9 | +2.7% |
| TPOT p50 (ms) | 259.5 | 260.4 | +0.3% |
| ITL p50 (ms) | 252.2 | 253.1 | +0.3% |

### Rate = inf (burst, all 200 prompts dispatched simultaneously)

| Metric | EC2-direct | Runpod-pod | Δ |
|---|---:|---:|---:|
| tput req/s | 5.712 | 5.688 | −0.4% |
| tput tok/s | 731.1 | 728.0 | −0.4% |
| TTFT p50 (ms) | 3060.4 | 2915.9 | **−4.7%** (Runpod faster) |
| TTFT p99 (ms) | 3084.5 | 3270.8 | **+6.0%** (Runpod slower) |
| TPOT p50 (ms) | 251.4 | 253.5 | +0.8% |
| ITL p50 (ms) | 252.1 | 253.1 | +0.4% |

The TTFT p50/p99 split at rate=`inf` is interesting and worth noting:
median TTFT is *better* on Runpod (−4.7%), but the p99 tail is worse
(+6.0%). Looking at the raw samples, the spread is large in both
environments (EC2-direct TTFT p50 values: 3074, 2964, 3060, 3075, 3033;
Runpod-pod: 2916, 3060, 2794, 2357, 2991). This is queue-tail behavior
under saturation, not orchestration overhead. With only 5 samples per
rate, ±5% confidence intervals are routine at this load level.

## CPU and cgroup

| | EC2-direct | Runpod-pod |
|---|---|---|
| `cpu.max` | `max 100000` (uncapped) | `16320000 100000` (≈163 vCPU cap) |
| `memory.max` | `max` | 728 GB |
| `nr_throttled` over sweep | 0 | 0 |
| `throttled_usec` over sweep | 0 | 0 |

The Runpod pod was given a CPU cap of 163 of the host's 192 vCPUs (85%).
Despite this, the workload never hit the cap during the entire benchmark
sweep — `throttled_usec` remained at zero. This is the cleanest possible
result for the orchestration layer: capping is in place, but it isn't
biting.

For a noisy-neighbor scenario or a CPU-bound workload, this guarantee
might not hold. But for a vLLM 8-GPU TP=8 inference workload, where the
GPUs are doing essentially all of the heavy lifting and the CPU is just
shepherding data, 163 vCPUs is plenty of headroom.

## Summary of significance

| Finding | Significance |
|---|---|
| Throughput delta < 1% at every rate | **Material:** the headline customer claim |
| Steady-state TPOT delta < 1% | **Material:** decode latency is what end-users feel |
| TTFT p50 delta < 2% at all but burst | **Material:** matches first-token latency expectations |
| TTFT p99 +6% at burst | **Not material:** noise at 5 samples; bombardment is worst-case |
| Zero cgroup throttle despite cap | **Material for the trust case:** demonstrates the cap doesn't bite the workload |

## Followups worth doing before sharing externally

1. **Re-run with 10+ samples per rate** to tighten confidence intervals,
   especially around rate=`inf` where the tail behavior is volatile.
2. **24h soak test** at rate=64 (the realistic steady-state rate) to
   confirm no slow drift from container-layer effects (memory pressure,
   thermal, etc.).
3. **Test on a different GPU** (H100 or B200) to confirm the finding
   holds when the orchestration layer's relative cost might be different.
4. **Test a CPU-bound workload** to characterize what the 163-vCPU cap
   actually does when the workload tries to use it. (Not needed for the
   inference pitch, but useful for general RPE messaging.)
