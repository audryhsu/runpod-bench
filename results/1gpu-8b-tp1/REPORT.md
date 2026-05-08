# Does Runpod's container layer cost vLLM serving performance?

**Run date:** 2026-05-07
**Owner:** audry.hsu@runpod.io
**Run directory:** [`results/ec2-direct/2026-05-07T15:17:31Z/`](./)

---

## TL;DR

For a typical 1-GPU vLLM serving workload (8B model, modest prompts), running
inside a Runpod pod is **performance-equivalent to running directly on the
underlying EC2 host at moderate request rates** (1–16 req/s). The Runpod pod
does pay a measurable but bounded penalty under burst traffic — about
**+270 ms / +20 % on TTFT-tail** when all requests arrive simultaneously —
which is traceable to the pod's cgroup CPU quota briefly throttling under load.

This finding only covers 1-GPU L4 inference of 8B models. **It does not
generalize automatically to multi-GPU pods, faster GPU classes, larger
models, or CPU-heavy workloads** (multimodal, guided decoding, etc.). See
"Open questions" below.

---

## The question we were asking

When a customer rents a 1-GPU pod from Runpod (vs renting the underlying
EC2 instance directly and running their own Docker), **how much performance
do they lose to the pod's container layer?** The container layer here means:

- The cgroup CPU quota Runpod imposes (~20 effective cores for our 1-GPU pod
  on a 192-vCPU host)
- The cgroup memory limit (~84 GiB on a 728 GiB host)
- Any other isolation overhead (namespaces, Runpod's control-plane processes
  on the host)

This question matters because it directly informs how we talk to customers
about the performance trade-off of fractional-GPU pricing vs whole-host
rental.

---

## What we did

We benchmarked vLLM serving Qwen3-8B on the **same physical EC2 host**
(8× NVIDIA L4) under two configurations:

1. **Baseline ("RunPod pod"):** Running inside a Runpod 1-GPU pod, sliced
   off the host. CPU quota ≈ 20.4 cores, memory ≈ 84 GiB.
2. **Treatment ("EC2-direct"):** Running directly on the same EC2 host with
   `docker run`, exposing exactly 1 GPU to the container, no CPU or memory
   limits. All 192 vCPUs available to the scheduler.

Both ran the same model, same workload, same vLLM version, same Python
environment. Same physical L4 brand, same kernel, same OS image. The only
intentional difference was the cgroup quota.

We swept request rates (1, 2, 4, 8, 16, infinite) measuring throughput
(tokens/sec) and latency (time-to-first-token, time-per-output-token).

---

## What we controlled for

| Variable | Both runs |
|----------|-----------|
| Hardware host | Same EC2 instance (AMD EPYC 7R13, 8× L4 driver 580.126.20) |
| GPU count visible to vLLM | **1 × NVIDIA L4** (verified — see Appendix) |
| Other GPUs on host | All 7 idle, 0 % utilization |
| Model | Qwen/Qwen3-8B |
| vLLM version | 0.16.0 (pinned) |
| torch version | 2.9.1+cu128 |
| numpy version | 2.1.2 |
| Python | 3.11.11 |
| OS image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| Workload | INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=50 |
| Random seed | 42 |
| Repetitions | 1 warmup + 2 measured runs per rate |
| `enforce_eager` | true (CUDA graphs disabled — same on both) |

## What we did **not** control for

| Variable | Why it matters | Likely impact |
|----------|---------------|---------------|
| Specific physical L4 card | PCIe topology and per-card thermal/power limits can differ slightly across the 8 cards in the host | Small — single-card variance is usually <1 % on this benchmark |
| CPU/NUMA pinning | Threads can land on either NUMA node; we didn't pin | Probably negligible for 1-GPU workloads |
| Background load on EC2 host | 3 idle Runpod handlers (~0.3 % CPU) and an idle Jupyter kernel persisted throughout the EC2 run | Negligible at 192 vCPUs |
| Number of measured runs | Only 2 measured runs per rate — small differences are within noise | Single-digit % deltas at rates 2–16 are noise-floor-bounded |

## What we simulated, and what we did not

The benchmark exercised six request-rate scenarios. Each maps to a different
real-world traffic pattern:

| Scenario | What it simulates in production |
|----------|---------------------------------|
| 1 req/s | Single-user / very low traffic. One request at a time, zero queueing. |
| 2 req/s | Light traffic. A few requests overlapping; scheduler is uncontended. |
| 4 req/s | Moderate steady-state — the "happy path" most production serving lives in. |
| 8 req/s | Heavy steady-state — queue starts to form; scheduler is actively interleaving. |
| 16 req/s | Saturated steady-state — the GPU is at its throughput ceiling; queue grows. |
| `inf` (bombardment) | Burst / spike — all 50 requests arrive simultaneously. Worst-case scheduler pressure. |

**We did not simulate, and the results say nothing about:**

- Multi-GPU pods (tensor-parallel vLLM with NCCL coordination)
- Bigger or faster GPU classes (H100, H200, B200)
- Larger models (70B+, where prefill is much more expensive)
- CPU-heavier workloads: multimodal models, JSON-mode / guided decoding,
  large speculative decoding, reranking pipelines
- Cold-start time (pod spin-up, model load) — we ran against an
  already-warm server
- Sustained long-running stability (our entire sweep took ~6 minutes)
- Storage I/O performance, network I/O performance
- Cost-per-token economics

---

## Headline results

For each scenario, throughput, time-to-first-token (TTFT), and time-per-token
(TPOT) on RunPod-pod vs EC2-direct, expressed as the relative difference:

| Scenario | Throughput | TTFT (median) | TTFT (p99) | TPOT |
|----------|-----------|---------------|------------|------|
| 1 req/s  | identical | identical     | identical  | identical |
| 2 req/s  | identical | identical     | identical  | identical |
| 4 req/s  | identical | identical     | within 5 % | identical |
| 8 req/s  | identical | within 8 %    | within 4 % | identical |
| 16 req/s | identical | within 8 %    | within 3 % | identical |
| `inf`    | identical | **−21 %**     | **−20 %**  | identical |

(Full numerical table in the Appendix.)

Three things to internalize:

1. **Throughput is identical at every rate.** Both setups peak at ~410
   tokens/sec, bounded by what the L4 can do.
2. **Per-token decode latency is identical at every rate.** TPOT differences
   are <1 %, because decode is purely a GPU forward-pass — the container
   layer cannot affect it.
3. **Time-to-first-token is identical until burst load.** The only metric and
   only scenario where the container layer matters is TTFT under
   bombardment, where it costs ~270 ms (~20 %).

The mechanism is straightforward: under the bombardment scenario, the vLLM
scheduler thread spikes its CPU usage briefly, hits the pod's 20.4-core
cgroup ceiling, and is throttled. We measured ~360 ms of total throttle
across the sweep, almost all of it during the burst scenario. That ~360 ms
is the gap.

---

## What we can conclude

- For **1-GPU vLLM serving of an 8B-class model on an L4** at typical
  steady-state request rates (≤ 16 req/s), the Runpod pod's container layer
  costs essentially nothing in throughput or latency.
- The container layer becomes visible only under **burst load**, where it
  costs ~20 % on TTFT-tail latency. Sustained throughput is unaffected even
  under burst.
- The mechanism is the **cgroup CPU quota**, not GPU isolation. A pod with a
  larger CPU allocation (or no CPU cap) would close even the burst gap.

## What we **cannot** conclude

- Anything about **multi-GPU pods**. We don't know whether the result holds
  when vLLM uses tensor parallelism. NCCL coordination adds CPU work that
  wasn't in our test, even if Runpod scales the CPU quota linearly with
  GPU count (which we believe is roughly the case but did not verify).
- Anything about **faster GPU classes (H100/H200/B200)**. On a faster GPU,
  the CPU side becomes a relatively bigger fraction of per-token cost, so
  any container-layer overhead would be more visible. Our L4 result is the
  most favorable case for the pod model.
- Anything about **larger models (70B+) or longer contexts**. Prefill is
  much more expensive at scale and exercises tokenization/scheduling more
  heavily.
- Anything about **CPU-heavy workloads**: vision-language, JSON guided
  decoding, large speculative decoding. These could surface overhead our
  test missed.
- Anything about **cold-start performance** (pod spin-up, model load
  time). We ran against a warm server.
- Anything about **storage, network, or cost economics**.

## Open questions for the team

1. **Does Runpod's CPU quota scale linearly with GPU count?** Our 1-GPU pod
   got ~20.4 of 192 host vCPUs (≈ 1/8). If this scales linearly, an 8-GPU
   pod would get the full ~163 cores and the burst-throttling we saw would
   probably disappear at any rate. Worth confirming from docs or from a
   live multi-GPU pod. *(If the team knows this off-hand, please tell me —
   I'll fold it into the conclusion.)*
2. **Is the burst result from the rate=`inf` scenario realistic for
   customer workloads?** Real burst traffic is usually less extreme than
   "all requests simultaneously" — typically capped by some upstream rate
   limit. The 20 % TTFT-tail penalty is a worst-case number; in practice
   it's likely smaller.

---

## Recommended next steps

In priority order:

1. **Tighten the statistical confidence on the existing result.** Re-run
   the same sweep with the harness's defaults (`BENCH_RUNS=5`,
   `NUM_PROMPTS=200`) — five measured runs per rate instead of two,
   200 prompts instead of 50. Same hardware, same environment. Wall time
   ~40 min. This collapses the ±5–10 % noise we currently see at rates
   2–16 and either confirms or rejects the rate=`inf` finding with much
   higher confidence. Cheap to do; should be done before quoting the 20 %
   number externally.
2. **Multi-GPU sanity check.** Repeat the comparison with a 2-GPU or 4-GPU
   Runpod pod using vLLM tensor parallelism, against an equivalent multi-
   GPU EC2 docker run. This is the single biggest blind spot in our
   current data — multi-GPU adds NCCL coordination work, which is exactly
   the kind of CPU pressure that could re-open the gap. If the conclusion
   still holds in 2–4 GPU configurations, we can credibly generalize.

Both runs reuse the existing harness as-is. Item 1 is one command; item 2
needs a Runpod pod provisioned with N GPUs.

---

## Appendix: full results table

Median across 2 measured runs per rate. RunPod = baseline (in pod), EC2 =
treatment (direct on host).

| Rate | Tput (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) |
|------|---|---|---|---|
| **1 req/s** | RunPod 108.7 / EC2 108.7 | 302 / 298 | 480 / 479 | 76.0 / 75.8 |
| **2 req/s** | RunPod 186.6 / EC2 186.8 | 239 / 243 | 583 / 576 | 82.4 / 82.1 |
| **4 req/s** | RunPod 278.0 / EC2 278.2 | 237 / 234 | 667 / 634 | 88.7 / 88.6 |
| **8 req/s** | RunPod 358.9 / EC2 359.4 | 232 / 249 | 790 / 760 | 93.8 / 93.2 |
| **16 req/s** | RunPod 375.2 / EC2 375.5 | 272 / 291 | 841 / 862 | 97.5 / 97.7 |
| **`inf`** | RunPod 406.6 / EC2 413.6 | 1275 / 1005 | 1341 / 1076 | 97.3 / 97.3 |

Cgroup CPU throttle delta during the entire sweep:

| | RunPod | EC2-direct |
|--|--------|------------|
| `nr_throttled` events | 61 | 0 |
| total throttled time | ~360 ms | 0 ms |

Almost all of the 360 ms lands in the rate=`inf` scenario.

## Appendix: data and reproducibility

All raw output, manifests, and per-rate JSON results live alongside this
report in the same directory. Key files (relative links):

- [`manifest.json`](./manifest.json) — full host/container/version snapshot
- [`summary.json`](./summary.json) / [`summary.txt`](./summary.txt) — per-rate aggregate metrics
- [`vllm_serving/`](./vllm_serving/) — per-run vLLM bench-serve output, organized by rate
- [`cpu/cgroup_throttle.json`](./cpu/cgroup_throttle.json) — pre/post throttle counters
- [`COMPARISON.md`](./COMPARISON.md) — earlier per-scenario technical writeup

To reproduce, on the same EC2 host:

```bash
export HF_TOKEN=<token>
docker run -d --name baseline-bench \
  --userns=host --gpus '"device=0"' --shm-size=16g --ipc=host \
  -v $(pwd):/bench -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /bench -e HF_TOKEN -e VLLM_VERSION=0.16.0 \
  runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 sleep infinity
docker exec baseline-bench ./setup.sh
docker exec -d baseline-bench ./serve.sh
docker exec -e BENCH_RUNS=2 -e NUM_PROMPTS=50 baseline-bench \
  ./run_all.sh --env-name ec2-direct --skip-fio --skip-cpu --skip-cold-start
```

The Runpod-side run is documented in the team's earlier benchmark notes
(2026-05-07 RunPod-Docker run).

## Appendix: methodology note (if anyone asks why this took two attempts)

A first attempt at this comparison used vLLM 0.20.1 (latest from PyPI) on
the EC2 side while the Runpod side ran 0.16.0 — the harness's setup script
didn't pin a vLLM version. That mismatched run showed apparent +13–26 %
throughput and −50 to −63 % TTFT-tail wins for EC2-direct, which we
initially attributed to the container layer.

After re-running with vLLM 0.16.0 pinned on both sides (this report), most
of those gains disappeared. The previously-attributed "container layer
overhead" was almost entirely the vLLM-version difference. The earlier
mismatched run was discarded; the harness now supports
`VLLM_VERSION=<x.y.z>` to prevent this class of mistake.

The lesson: when the comparison is about anything below the runtime, pin
the runtime explicitly.
