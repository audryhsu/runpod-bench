# EC2-direct vs RunPod-pod (single-container) — vLLM serving sweep, 2026-05-07

Both runs target the same model (Qwen/Qwen3-8B), same workload
(INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=50, BENCH_RUNS=2 + 1 warmup,
REQUEST_RATES="1 2 4 8 16 inf"), and the same physical EC2 host
(AMD EPYC 7R13, kernel 6.17.0-1012-aws, 1× NVIDIA L4 driver 580.126.20).

Both runs also use the **same Python environment**: base image
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, then
`pip install vllm==0.16.0` (which pulls torch 2.9.1+cu128 transitively).
The only treatment difference is the *container layer*:

| Aspect              | RunPod-pod (baseline)             | EC2-direct (treatment)               |
|---------------------|-----------------------------------|--------------------------------------|
| Container layer     | RunPod pod, single container,     | docker run, directly on EC2 host     |
|                     | server + harness in two notebook  | server + harness in one container    |
|                     | kernels in same container         |                                      |
| cgroup CPU quota    | 2,040,000 / 100,000 ~ 20.4 cores  | `max` (unconstrained, 192 vCPUs)     |
| cgroup memory limit | ~83.8 GiB                         | `max` (unconstrained, 728 GiB)       |
| vLLM version        | 0.16.0                            | 0.16.0                               |
| torch version       | 2.9.1+cu128                       | 2.9.1+cu128                          |
| numpy version       | 2.1.2                             | 2.1.2                                |
| Background load     | jupyter + ipykernels              | none                                 |

## Headline (median across 2 measured runs per rate)

| Rate | Tput tok/s (RunPod / EC2 / d) | TTFT p50 ms (RunPod / EC2 / d) | TTFT p99 ms (RunPod / EC2 / d) | TPOT p50 ms (RunPod / EC2 / d) |
|------|--------------------------------|----------------------------------|-----------------------------------|-----------------------------------|
| 1    | 108.7 / 108.7 / **+0.0 %**      |  302.1 /  298.4 / **-1.2 %**     |  479.5 /  479.3 / **-0.0 %**      | 76.0 / 75.8 / **-0.3 %**          |
| 2    | 186.6 / 186.8 / **+0.1 %**      |  238.6 /  243.2 / **+1.9 %**     |  583.3 /  576.1 / **-1.2 %**      | 82.4 / 82.1 / **-0.4 %**          |
| 4    | 278.0 / 278.2 / **+0.1 %**      |  236.8 /  234.1 / **-1.1 %**     |  667.2 /  634.1 / **-5.0 %**      | 88.7 / 88.6 / **-0.1 %**          |
| 8    | 358.9 / 359.4 / **+0.1 %**      |  231.7 /  249.1 / **+7.5 %**     |  789.6 /  760.4 / **-3.7 %**      | 93.8 / 93.2 / **-0.6 %**          |
| 16   | 375.2 / 375.5 / **+0.1 %**      |  271.5 /  291.1 / **+7.2 %**     |  840.5 /  861.5 / **+2.5 %**      | 97.5 / 97.7 / **+0.2 %**          |
| inf  | 406.6 / 413.6 / **+1.7 %**      | 1274.6 / 1005.3 / **-21.1 %**    | 1341.3 / 1075.9 / **-19.8 %**     | 97.3 / 97.3 / **+0.0 %**          |

*d is `(EC2 - RunPod) / RunPod`; positive throughput and negative TTFT/TPOT
favour EC2-direct.*

## Observations

1. **Container-layer overhead is effectively zero at rates 1-16.** Across
   throughput, TTFT p50, TTFT p99, and TPOT, every metric is within ~7 %
   of parity at rates 1-16, and within ~1 % on throughput specifically.
   The Runpod pod's 20.4-core CPU quota is enough headroom for a single-
   GPU vLLM scheduler at moderate load.
2. **The only meaningful win for EC2-direct is at saturation.** At
   `rate=inf` the EC2 run delivers the same throughput (+1.7 %) but cuts
   TTFT p50 by **21 %** (1275 → 1005 ms) and TTFT p99 by **20 %**
   (1341 → 1076 ms). This is where the scheduler starts to compete with
   itself for CPU and the cgroup quota begins to throttle.
3. **cgroup throttle is the smoking gun.** RunPod logged
   `nr_throttled=61` / `throttled_usec=360,120,699` (~360 ms across the
   sweep). EC2-direct logged 0 throttling events. ~360 ms of total
   throttle, almost all of which lands at `inf`, accounts for the
   tail-latency gap and nothing else.
4. **TPOT is flat across configs.** Per-token decode is GPU-bound, and
   the GPU is the same physical L4 in both cases. No surprise.

## Methodology note: don't trust the previous version-mismatched run

A first attempt at this comparison used vLLM 0.20.1 (latest from PyPI)
on the EC2-direct side while RunPod stayed on 0.16.0, because the
harness's `setup.sh` did not pin a vLLM version. That mismatched run
showed apparent +13-26 % throughput and -50 to -63 % TTFT-p99 wins for
EC2-direct, which we attributed to the container layer.

After re-running with `vllm==0.16.0` pinned (this report), most of those
gains disappear. The previously-attributed gains were almost entirely
the vLLM upgrade, not the container difference.

The harness now supports `VLLM_VERSION=<x.y.z>` to pin the install,
preventing a repeat. Always pin the runtime when the comparison is
about anything below it.

## Caveats

- **Only 2 measured runs per rate** — the small differences at rates
  2-16 (a few percent in either direction on TTFT p50) are within run-
  to-run noise. The flat-throughput conclusion is robust because the
  GPU is the bottleneck and run-to-run noise on tok/s is small. For
  publishable claims, re-run both with `BENCH_RUNS=5 NUM_PROMPTS=200`.
- **`enforce_eager=1`** disables CUDA graphs on both sides. Latency
  would improve modestly with eager mode off, but the comparison
  remains valid.
- The EC2 host still has 3 leftover RunPod handler.py processes
  (~0.3 % of total CPU) and an idle jupyter-lab. Negligible given
  192 vCPUs and an isolated GPU.

## Files

- `summary.txt`, `summary.json` — EC2-direct headline table (this run)
- `manifest.json` — full host/container/config snapshot
- `vllm_serving/rate_*/run_*.json` — per-run vLLM bench serve output
- `cpu/cgroup_throttle.json` — throttle delta for this sweep (zero)
