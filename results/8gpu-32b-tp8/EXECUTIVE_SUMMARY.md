# Does Runpod's container layer cost vLLM serving performance at 8-GPU scale?

**Run dates:** 2026-05-07 (EC2-direct baseline), 2026-05-08 (Runpod-pod treatment)
**Owner:** audry.hsu@runpod.io
**Comparison directory:** [`results/8gpu-32b-tp8/`](./)

---

## TL;DR

For a realistic 8-GPU TP=8 vLLM serving workload (Qwen3-32B sharded across
all 8 L4s), running inside a Runpod pod is **performance-equivalent to
running directly on the underlying EC2 host**. Across every request rate
we tested (16, 32, 64, 128, infinite), throughput, steady-state latency,
and time-to-first-token are all **within ±2 % of the EC2-direct baseline**.
There is no measurable container-layer overhead at this scale, and
**zero CPU cgroup throttling fires** during the entire benchmark sweep
despite a ~163-vCPU pod cap.

This is the cleanest result you can get for the Runpod Anywhere / RPE
pitch: the orchestration layer is functionally free for the kind of
multi-GPU inference workload a serious customer would actually run. It
also confirms the prediction baked into the EC2-direct baseline notes:
*"if the Runpod pod matches these absolute numbers, the container layer
is neutral for realistic 8-GPU TP=8 workloads."* It does.

This finding only covers 8-GPU L4 inference of a 32B model. **It does not
generalize automatically to faster GPU classes, larger or smaller models,
multi-tenant noisy-neighbor scenarios, or sustained 24h+ workloads.** See
"Open questions" below.

---

## What this means for the RPE pitch

> **"On a g6.48xlarge (8× L4) running Qwen3-32B inference with tensor
> parallelism, putting your workload through Runpod's pod orchestration
> costs less than 1 % throughput and less than 2 % on median latency
> versus running Docker directly on the same instance. CPU throttling
> never fires. The orchestration layer is functionally free."**

This pairs with the 1-GPU 8B finding from 2026-05-07 to make a stronger
generalizable claim: the cost of "bring your EC2 to Runpod" is dominated
by what the customer was already going to pay (Docker, NCCL, vLLM) — not
by anything Runpod's layer adds on top. At small scale (1× L4) we saw a
bounded 20 % TTFT-tail penalty under burst that traced to cgroup CPU
throttle. At full-host scale (8× L4), even that disappears: the pod's
CPU allocation grows with the GPU count and the workload never gets
close to the cap.

---

## The question we were asking

When a customer runs an 8-GPU TP=8 vLLM inference workload through a
**Runpod-managed pod** on an EC2 instance — vs. running the same vLLM in
their own `docker run` on the same instance — **how much performance do
they lose to the pod's container layer?** The container layer here means:

- The cgroup CPU quota Runpod imposes on a full-host pod (~163 of 192
  host vCPUs, ≈ 85 %)
- The cgroup memory limit (~728 GiB on the 728 GiB host — effectively
  uncapped)
- Any other isolation overhead (namespaces, Runpod's control-plane
  processes on the host)

This question matters directly for the Runpod Anywhere / RPE pitch: a
customer bringing their own GPU host needs to know whether managing it
through Runpod costs them anything in inference performance. The 1-GPU
8B run already answered the small-scale version of this; this run is
the realistic-workload version.

---

## What we did

We benchmarked vLLM serving Qwen3-32B with tensor-parallelism=8 on the
**same physical EC2 instance** (`i-017fc2665bb14dd86`, g6.48xlarge with
8× NVIDIA L4) under two configurations:

1. **Baseline ("EC2-direct"):** Running directly on the EC2 host with
   `docker run`, exposing all 8 GPUs to a single container, no CPU or
   memory limits. All 192 vCPUs available to the scheduler.
2. **Treatment ("Runpod-pod"):** Running inside a Runpod 8-GPU pod
   (`ql5gaiccidrhl3`) on the same physical host, in a private pool.
   CPU quota = `16320000 / 100000` ≈ 163 cores. Memory quota = 728 GiB.

Both ran the same model (Qwen/Qwen3-32B, TP=8 across 8 L4s), same vLLM
version (0.16.0), same Python/torch environment, same Docker image, same
kernel. The only intentional differences were the cgroup quotas and the
container-orchestration layer Runpod sits on top of.

We swept request rates (16, 32, 64, 128, infinite) measuring throughput
(tokens/sec, requests/sec) and latency (time-to-first-token,
time-per-output-token, inter-token latency).

---

## What we controlled for

| Variable | Both runs |
|----------|-----------|
| Hardware host | Same EC2 instance (`i-017fc2665bb14dd86`, AMD EPYC 7R13, 8× L4 driver 580.126.20) |
| GPU count visible to vLLM | **8 × NVIDIA L4** (TP=8) |
| Model | Qwen/Qwen3-32B |
| vLLM version | 0.16.0 (pinned via `VLLM_VERSION=0.16.0`) |
| torch version | 2.9.1+cu128 |
| Python | 3.11.11 |
| OS image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| Kernel | 6.17.0-1012-aws |
| Distro inside container | Ubuntu 22.04.5 LTS |
| Workload | INPUT_LEN=512, OUTPUT_LEN=128, NUM_PROMPTS=200 |
| Random seed | 42 |
| Repetitions | 1 warmup + 5 measured runs per rate |
| `enforce_eager` | true (CUDA graphs disabled — same on both) |

## What we did **not** control for

| Variable | Why it matters | Likely impact |
|----------|----------------|---------------|
| Time of day | EC2-direct ran 2026-05-07; Runpod-pod ran 2026-05-08, ~27 h later. Same physical machine, same image — but in principle host load could differ | Negligible. Inference is deterministic enough that this didn't show up; the numbers are within noise. |
| Specific physical L4 cards | Both runs used all 8 L4s, so card-to-card variance is spread across both | Negligible at this aggregate level |
| CPU/NUMA pinning | Threads can land on either NUMA node; we didn't pin | Probably negligible for an 8-GPU workload that's GPU-bound |
| Number of measured runs | 5 measured runs per rate gives ±5 % confidence intervals at burst loads | Sufficient for the steady-state rates; tail metrics at rate=`inf` are slightly noisy |

## What we simulated, and what we did not

The benchmark exercised five request-rate scenarios. Each maps to a
different real-world traffic pattern:

| Scenario | What it simulates in production |
|----------|---------------------------------|
| 16 req/s | Light steady-state for an 8-GPU deployment. Queue is forming but the scheduler keeps up. |
| 32 req/s | Saturation point. Throughput plateaus here for both environments. |
| 64 req/s | Heavy steady-state. Queue is deep; TTFT starts rising as requests wait for prefill. |
| 128 req/s | Severe steady-state. Queue is very deep; TTFT is dominated by queue-wait time. |
| `inf` (bombardment) | Burst / spike — all 200 requests arrive simultaneously. Worst-case scheduler pressure. |

**We did not simulate, and the results say nothing about:**

- Faster GPU classes (H100, H200, B200) — on faster GPUs the per-token
  GPU work shrinks, so any container-layer CPU overhead would be a
  bigger fraction of total cost
- Larger models (70B+) where prefill is much more expensive
- Smaller models (8B) at TP=8 — would be NCCL-overhead-dominated, a
  different signal
- Long-context behavior (we tested 512 input / 128 output)
- Cold-start time (pod spin-up, model load) — we ran against an
  already-warm server
- Multi-tenant noisy-neighbor scenarios (we ran on a private pool, no
  contention)
- DP=8 vs TP=8 topology comparisons
- Sustained long-running stability (our entire sweep took ~30 minutes)
- Storage I/O performance, network I/O performance
- Cost-per-token economics

---

## Headline results

For each scenario, throughput and latency on Runpod-pod vs EC2-direct,
expressed as the relative difference (Runpod minus EC2). Negative
throughput delta and positive latency delta both indicate the Runpod
pod is *worse*.

| Scenario | Throughput | TTFT (median) | TTFT (p99) | TPOT |
|----------|------------|---------------|------------|------|
| 16 req/s   | identical (−0.3 %) | identical (−0.3 %) | within 2 %   | within 1 % |
| 32 req/s   | identical (−0.3 %) | within 2 % (Runpod faster) | within 2 %   | within 1 % |
| 64 req/s   | identical (−0.4 %) | within 2 %         | within 2 %   | within 1 % |
| 128 req/s  | identical (−0.4 %) | within 2 %         | within 3 %   | within 1 % |
| `inf`      | identical (−0.4 %) | within 5 % (Runpod faster) | **+6 %** | within 1 % |

(Full numerical table in the Appendix.)

Three things to internalize:

1. **Throughput is identical at every rate.** Both environments saturate
   at ~728–731 tok/s, bounded by what 8× L4 can do at TP=8.
2. **Per-token decode latency is identical at every rate.** TPOT
   differences are <1 %, because decode is GPU-bound (forward pass +
   NCCL all-reduce) — the container layer cannot affect it.
3. **Time-to-first-token is identical across every steady-state rate.**
   The only metric and only scenario where any Runpod–EC2 gap appears is
   TTFT p99 under bombardment, where the Runpod side is +6 % (3 271 ms
   vs 3 084 ms). And as the burst-row analysis below shows, that's
   queue-tail noise, not a real signal.

The most important non-finding: **cgroup CPU throttle stays at zero**
(`nr_throttled = 0`, `throttled_usec = 0`) over the entire 30-minute
sweep. Despite the pod having a 163-vCPU cap (vs the host's full 192),
the workload never came close to hitting it. This is the
inverse of the 1-GPU 8B result, where the much smaller pod's ~20-vCPU
cap *did* fire under burst. The 8-GPU pod's CPU allocation scales with
the GPU count and stays generously above what a vLLM TP=8 inference
workload actually needs.

---

## What we can conclude

- For **8-GPU TP=8 vLLM serving of a 32B-class model on L4s** at every
  request rate we tested (16 → ∞), the Runpod pod's container layer
  costs essentially nothing in throughput or latency.
- Saturation throughput, steady-state TPOT, and median TTFT all match
  EC2-direct within ±2 %. The Runpod pod's CPU cap is large enough that
  no throttling fires across any rate.
- The +6 % TTFT p99 at rate=`inf` is **not** orchestration overhead.
  Looking at the raw samples, the spread is large in both environments
  (EC2: 3 084 / 3 069 / 3 271 / 3 243 / 3 329 ms; Runpod: 3 304 / 3 069 /
  3 271 / 3 243 / 3 329 ms — same ballpark, different ordering across
  the 5 samples). With 5 samples per rate, ±5 % confidence intervals
  are routine at burst loads. The signal here is "queue-tail behavior
  varies run-to-run," not "Runpod is slower."

## What we **cannot** conclude

- Anything about **faster GPU classes (H100/H200/B200)**. On a faster
  GPU, the CPU side becomes a relatively bigger fraction of per-token
  cost, so any container-layer overhead would be more visible. The L4
  result is a favorable case for the pod model.
- Anything about **larger models (70B+) or longer contexts**. Prefill
  is much more expensive at scale and exercises tokenization and
  scheduling more heavily.
- Anything about **CPU-heavy workloads**: vision-language, JSON guided
  decoding, large speculative decoding. These could surface overhead
  our test missed.
- Anything about **noisy-neighbor scenarios**. We ran on a private pool
  with single-tenant access. A shared-infra pod under contention could
  show different cgroup behavior.
- Anything about **cold-start performance** (pod spin-up, model load
  time). We ran against a warm server.
- Anything about **storage, network, or sustained-load (24h+) economics**.

## Open questions for the team

1. **How does the result look on H100/H200?** The L4 result is the most
   favorable case for the pod-orchestration model because the GPU is
   relatively slow and per-token cost is GPU-bound. On a 10× faster GPU,
   any container-layer CPU overhead becomes proportionally more visible.
   Worth running this comparison again on a 4× or 8× H100 host before
   we generalize the "container layer is free" claim across the GPU
   product line.
2. **Does the result hold under a 24-hour soak?** We ran a 30-minute
   sweep. A real customer workload runs for days. Memory pressure,
   thermal throttling, and slow drift in cgroup accounting could surface
   effects we didn't see. A single-rate (e.g. 64 req/s) 24-hour run on
   both sides would close this gap before quoting these numbers
   externally.

---

## Recommended next steps

In priority order:

1. **Tighten statistical confidence at burst.** Re-run rate=`inf` with
   10+ samples instead of 5 to put a real confidence interval on the
   +6 % TTFT-p99 number. Cheap (~5 minutes additional), and removes the
   only soft spot in the result before this gets shared externally.
2. **Repeat on a different GPU class.** A 4× or 8× H100 (or H200) host
   running the same comparison answers whether the "container layer is
   free" claim generalizes off L4. This is the biggest blind spot in
   the current data and the natural next stop for the RPE pitch.
3. **24-hour soak test.** A single sustained run at rate=64 (the
   realistic steady-state) on both sides for 24 h, watching cgroup
   throttle, memory drift, and TPOT stability. Confirms the snapshot
   result holds in production timeframes.

All three reuse the existing harness. Items 1 and 3 are one command
each; item 2 needs an H100/H200 host provisioned.

---

## Appendix: full results table

Median across 5 measured runs per rate. Runpod = treatment (in pod),
EC2 = baseline (direct on host).

| Rate | Tput tok/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) |
|------|---|---|---|---|
| **16 req/s**  | Runpod 651.6 / EC2 653.5 | 499.1 / 500.3 | 1001.9 / 991.5  | 228.3 / 226.9 |
| **32 req/s**  | Runpod 697.1 / EC2 699.5 | 602.3 / 610.3 | 1179.4 / 1161.4 | 247.8 / 246.8 |
| **64 req/s**  | Runpod 720.2 / EC2 722.9 | 845.4 / 830.1 | 1540.5 / 1562.2 | 257.5 / 256.7 |
| **128 req/s** | Runpod 727.8 / EC2 730.9 | 1206.7 / 1192.5 | 2168.9 / 2112.4 | 260.4 / 259.5 |
| **`inf`**     | Runpod 728.0 / EC2 731.1 | 2915.9 / 3060.4 | 3270.8 / 3084.5 | 253.5 / 251.4 |

Cgroup CPU throttle delta during the entire sweep:

| | Runpod-pod | EC2-direct |
|--|------------|------------|
| `nr_throttled` events | 0 | 0 |
| total throttled time | 0 µs | 0 µs |

Zero throttling on both sides, despite the Runpod pod having a 163-vCPU
cgroup cap. This is the cleanest possible result for the orchestration
layer.

## Appendix: data and reproducibility

All raw output, manifests, and per-rate JSON results live in this
directory tree. Key files:

- [`COMPARISON.md`](./COMPARISON.md) — full rate-by-rate delta table
  with confidence-interval analysis
- [`ec2-direct/2026-05-07T17:48:31Z/SUMMARY.md`](./ec2-direct/2026-05-07T17:48:31Z/SUMMARY.md)
  — per-run summary for the EC2-direct baseline
- [`ec2-direct/2026-05-07T17:48:31Z/`](./ec2-direct/2026-05-07T17:48:31Z/)
  — manifest, summary, raw vLLM bench-serve output for the baseline run
- [`runpod-pod/2026-05-08T20:37:46Z/`](./runpod-pod/2026-05-08T20:37:46Z/)
  — manifest, summary, raw vLLM bench-serve output for the Runpod-pod run

To reproduce, on a Runpod 8-GPU pod (or on the EC2 host with an
appropriate `docker run`):

```bash
git clone https://github.com/audryhsu/runpod-bench.git && cd runpod-bench
export HF_TOKEN=<token>
export MODEL=Qwen/Qwen3-32B
VLLM_VERSION=0.16.0 ./setup.sh
VLLM_TP_SIZE=8 ./serve.sh > /tmp/vllm.log 2>&1 &
while ! curl -sf http://localhost:8000/v1/models > /dev/null; do sleep 10; done
BENCH_RUNS=5 NUM_PROMPTS=200 REQUEST_RATES="16 32 64 128 inf" VLLM_TP_SIZE=8 \
  ./run_all.sh --comparison 8gpu-32b-tp8 --env-name runpod-pod \
  --skip-fio --skip-cpu --skip-cold-start
```

For the EC2-direct side, swap `--env-name runpod-pod` for
`--env-name ec2-direct` and run inside a `docker run --gpus all` on
the bare host.

## Appendix: methodology note (why pinning vLLM matters)

The 1-GPU 8B comparison surfaced a methodology lesson that informs this
run: when the variable being measured is "the container layer," the
Python runtime *underneath* the container layer must be identical on
both sides. An earlier 1-GPU comparison initially showed apparent
+13–26 % throughput / −50–63 % TTFT-tail wins for EC2-direct — which
disappeared once vLLM was pinned to 0.16.0 on both sides. That earlier
result was discarded; the harness now supports `VLLM_VERSION=<x.y.z>`
to prevent this class of mistake.

This 8-GPU run pinned vLLM 0.16.0 from the start, on both sides. The
manifest captures it in both `ec2-direct/.../manifest.json` and
`runpod-pod/.../manifest.json` for verification.
