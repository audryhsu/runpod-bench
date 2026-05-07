# EC2-direct vs RunPod-Docker — vLLM serving sweep, 2026-05-07

Both runs target the same model (Qwen/Qwen3-8B), same workload
(INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=50, BENCH_RUNS=2 + 1 warmup,
REQUEST_RATES="1 2 4 8 16 inf"), and the same physical EC2 host
(AMD EPYC 7R13, kernel 6.17.0-1012-aws, 1× NVIDIA L4 driver 580.126.20).

The treatment difference is the container layer:

| Aspect              | RunPod-Docker (baseline)    | EC2-direct (treatment)               |
|---------------------|-----------------------------|--------------------------------------|
| Container layer     | RunPod pod -> docker        | docker, run directly on EC2 host     |
| cgroup CPU quota    | 2,040,000 / 100,000 ~ 20.4 cores | `max` (unconstrained, 192 vCPUs) |
| cgroup memory limit | ~83.8 GiB                   | `max` (unconstrained, 728 GiB)       |
| vLLM version        | 0.16.0                      | 0.20.1                               |
| torch version       | 2.9.1+cu128                 | 2.11.0+cu130                         |
| Background load     | jupyter + ipykernels in pod | none in this container               |
| Runpod handlers     | n/a (not in pod)            | left running on host (~0.3% CPU)     |

## Headline (median across 2 measured runs per rate)

| Rate | Tput tok/s (RunPod / EC2 / d) | TTFT p50 ms (RunPod / EC2 / d) | TTFT p99 ms (RunPod / EC2 / d) | TPOT p50 ms (RunPod / EC2 / d) |
|------|--------------------------------|----------------------------------|-----------------------------------|-----------------------------------|
| 1    | 108.7 / 108.0 / **-0.6 %**      | 302.1 / 193.0 / **-36.1 %**      |  479.5 /  229.8 / **-52.1 %**     | 76.0 / 73.0 / **-3.9 %**          |
| 2    | 186.6 / 184.1 / **-1.3 %**      | 238.6 / 210.3 / **-11.9 %**      |  583.3 /  260.9 / **-55.3 %**     | 82.4 / 80.9 / **-1.8 %**          |
| 4    | 278.0 / 275.5 / **-0.9 %**      | 236.8 / 228.0 / **-3.7 %**       |  667.2 /  283.3 / **-57.5 %**     | 88.7 / 87.6 / **-1.2 %**          |
| 8    | 358.9 / 359.3 / **+0.1 %**      | 231.7 / 227.2 / **-1.9 %**       |  789.6 /  305.8 / **-61.3 %**     | 93.8 / 91.8 / **-2.1 %**          |
| 16   | 375.2 / 424.6 / **+13.2 %**     | 271.5 / 229.9 / **-15.3 %**      |  840.5 /  311.8 / **-62.9 %**     | 97.5 / 93.6 / **-4.0 %**          |
| inf  | 406.6 / 511.3 / **+25.8 %**     | 1274.6 / 393.3 / **-69.1 %**     | 1341.3 /  517.5 / **-61.4 %**     | 97.3 / 94.9 / **-2.5 %**          |

*d is `(EC2 - RunPod) / RunPod`; negative TTFT/TPOT and positive throughput
favour EC2-direct.*

## Observations

1. **Throughput at low rates is GPU-bound, not container-bound.** At 1-8
   req/s the two configs are within 1.3 % of each other on tok/s — the L4
   is the bottleneck and the container layer doesn't matter.
2. **Saturation throughput is much higher on EC2-direct.** At the
   knee (rate=16) and at `inf` the EC2-direct run delivers
   **+13 %** and **+26 %** more tokens/sec respectively. This is the
   regime where the vLLM scheduler/tokenizer becomes CPU-bound and the
   RunPod pod's 20.4-core quota starts to throttle.
3. **TTFT p99 is dramatically lower on EC2-direct at every rate** (-52 %
   to -63 %). On RunPod it climbs steadily from 480 ms (rate 1) to
   1341 ms (inf); on EC2-direct it stays under 320 ms until rate `inf`.
   Tail latency is the most sensitive signal of head-of-line blocking
   from CPU contention in the request scheduler.
4. **TPOT p50 is essentially flat** across configs (-2 % to -4 %), which
   is consistent with TPOT being dominated by per-token GPU forward passes
   once a request is on the GPU.
5. **cgroup throttle**: RunPod hit `nr_throttled=61` /
   `throttled_usec=360,120,699` (~360 ms) during the sweep. EC2-direct
   recorded 0 throttling events. This is the underlying mechanism for
   the TTFT-tail and saturation-throughput gaps.

## Caveats

- **vLLM version differs (0.16.0 vs 0.20.1)** — the EC2-direct image
  installs fresh from PyPI, while the RunPod run reused a pre-built
  vLLM container. v0.20.1 includes scheduler and CUDA-graph
  improvements (we set `--enforce-eager` so CUDA-graph is disabled
  on both, but other scheduler changes still apply). Some of the
  high-rate throughput gain is likely attributable to vLLM itself,
  not to the container layer alone.
- **Only 2 measured runs per rate** — these numbers are noisier than a
  full default sweep (`BENCH_RUNS=5 NUM_PROMPTS=200`). For publishable
  claims, re-run both with the larger config.
- **`enforce_eager=1`** disables CUDA graphs on both sides. Latency
  would improve modestly with eager mode off, but the comparison
  remains valid.
- The EC2 host still has 3 leftover RunPod handler.py processes
  (~0.3 % of total CPU) and an idle jupyter-lab. These are negligible
  given 192 vCPUs and an isolated GPU, but noted for completeness.

## Files

- `summary.txt`, `summary.json` — EC2-direct headline table (this run)
- `manifest.json` — full host/container/config snapshot for this run
- `vllm_serving/rate_*/run_*.json` — per-run vLLM bench serve output
- `cpu/cgroup_throttle.json` — throttle delta for this sweep (zero)
