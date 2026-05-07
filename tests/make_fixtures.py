#!/usr/bin/env python3
"""Generate synthetic fixture JSON files for both ec2-direct and runpod-pod environments.

Mirrors the exact JSON shapes produced by:
  - vllm bench serve --save-result
  - fio --output-format=json
  - bench_cold_start.sh
  - bench_cpu.sh / sysbench
  - capture_manifest.sh
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).parent / "fixtures"

# ── helpers ────────────────────────────────────────────────────────────────────

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path.relative_to(BASE.parent)}")


def vllm_run(req_tput, tok_tput, ttft_med, ttft_p99, tpot_med, tpot_p99, itl_med, itl_p99):
    """Synthetic vllm bench serve --save-result JSON."""
    return {
        "request_throughput": req_tput,
        "output_throughput": tok_tput,
        "mean_ttft_ms": ttft_med * 1.05,
        "median_ttft_ms": ttft_med,
        "p99_ttft_ms": ttft_p99,
        "mean_tpot_ms": tpot_med * 1.03,
        "median_tpot_ms": tpot_med,
        "p99_tpot_ms": tpot_p99,
        "mean_itl_ms": itl_med * 1.02,
        "median_itl_ms": itl_med,
        "p99_itl_ms": itl_p99,
    }


def fio_result(bw_bytes, iops, lat_p50_ns, lat_p99_ns, rw):
    """Synthetic fio --output-format=json output."""
    io_block = {
        "bw_bytes": bw_bytes,
        "bw": bw_bytes // 1024,
        "iops": iops,
        "clat_ns": {
            "percentile": {
                "50.000000": lat_p50_ns,
                "99.000000": lat_p99_ns,
            }
        },
    }
    job = {"read": {}, "write": {}}
    job[rw] = io_block
    return {"jobs": [job]}


def cold_start_run(duration_s, run_num):
    start = 1746528000.0 + run_num * 45
    return {
        "start_epoch": start,
        "ready_epoch": start + duration_s,
        "duration_s": duration_s,
        "run": run_num,
    }


def manifest(env_name, cgroup_cpu_quota="max", vllm_version="0.8.3"):
    return {
        "hostname": f"ip-10-0-1-{42 if env_name == 'ec2-direct' else 99}",
        "in_container": True,
        "kernel": "6.8.0-1021-aws",
        "distro": "Ubuntu 24.04.1 LTS",
        "gpu_model": "NVIDIA L4",
        "gpu_count": 8,
        "gpu_driver": "580.159.03",
        "cuda_version": "12.8",
        "cpu_model": "Intel(R) Xeon(R) Platinum 8488C",
        "cpu_count": 192,
        "memory_gb": 768,
        "vllm_version": vllm_version,
        "torch_version": "2.8.0+cu128",
        "python_version": "3.11.9",
        "cgroup_cpu_quota": cgroup_cpu_quota,
        "cgroup_memory_limit": "max",
        "config_snapshot": {
            "model": "Qwen/Qwen3-8B",
            "vllm_port": "8000",
            "gpu_mem_util": "0.95",
            "max_model_len": "8128",
            "dtype": "auto",
            "enforce_eager": "1",
            "input_len": "512",
            "output_len": "128",
            "num_prompts": "200",
            "request_rates": "1 4 inf",
            "bench_runs": "5",
            "warmup_runs": "1",
            "seed": "42",
        },
        "timestamp": "2026-05-06T12:00:00Z",
    }


# ── ec2-direct fixtures ─────────────────────────────────────────────────────────

EC2 = BASE / "ec2-direct" / "2026-05-06T12:00:00Z"

print("Generating ec2-direct fixtures...")

# manifest
write(EC2 / "manifest.json", manifest("ec2-direct", cgroup_cpu_quota="max"))

# vLLM serving -- rate 1 (warmup run_0 + 5 measured runs)
# run_0 is warmup, slightly noisy
write(EC2 / "vllm_serving/rate_1/run_0.json",
      vllm_run(1.0, 128.0, 42.0, 58.0, 8.2, 10.1, 8.0, 9.5))  # warmup, discarded
for i, (ttft, tpot) in enumerate([(42.1, 8.2), (41.9, 8.1), (42.3, 8.3), (42.0, 8.2), (41.8, 8.1)], start=1):
    write(EC2 / f"vllm_serving/rate_1/run_{i}.json",
          vllm_run(1.0, 128.0, ttft, 57.0, tpot, 10.0, 8.0, 9.4))

# vLLM serving -- rate 4
write(EC2 / "vllm_serving/rate_4/run_0.json",
      vllm_run(4.0, 512.0, 55.0, 80.0, 8.5, 11.0, 8.3, 10.0))  # warmup
for i, (ttft, tpot) in enumerate([(55.2, 8.5), (54.8, 8.4), (55.5, 8.6), (55.0, 8.5), (54.9, 8.4)], start=1):
    write(EC2 / f"vllm_serving/rate_4/run_{i}.json",
          vllm_run(4.0, 512.0, ttft, 79.0, tpot, 10.8, 8.3, 9.8))

# vLLM serving -- rate inf
write(EC2 / "vllm_serving/rate_inf/run_0.json",
      vllm_run(18.5, 2368.0, 320.0, 450.0, 9.1, 14.0, 8.9, 13.0))  # warmup
for i, (ttft, tpot) in enumerate([(318.0, 9.0), (322.0, 9.1), (319.0, 9.0), (321.0, 9.1), (320.0, 9.0)], start=1):
    write(EC2 / f"vllm_serving/rate_inf/run_{i}.json",
          vllm_run(18.5, 2368.0, ttft, 448.0, tpot, 13.8, 8.9, 12.8))

# cold start (run_0 warmup + runs 1-5)
write(EC2 / "cold_start/run_0.json", cold_start_run(29.5, 0))  # warmup
for i, dur in enumerate([28.1, 28.3, 27.9, 28.5, 28.2], start=1):
    write(EC2 / f"cold_start/run_{i}.json", cold_start_run(dur, i))

# fio
write(EC2 / "fio/seq_read.json",
      fio_result(bw_bytes=3355443200, iops=3200, lat_p50_ns=300000, lat_p99_ns=1200000, rw="read"))
write(EC2 / "fio/seq_write.json",
      fio_result(bw_bytes=2097152000, iops=2000, lat_p50_ns=480000, lat_p99_ns=1800000, rw="write"))
write(EC2 / "fio/rand_read.json",
      fio_result(bw_bytes=1073741824, iops=4096, lat_p50_ns=950000, lat_p99_ns=3200000, rw="read"))
write(EC2 / "fio/rand_write.json",
      fio_result(bw_bytes=67108864, iops=16384, lat_p50_ns=220000, lat_p99_ns=800000, rw="write"))

# cpu
write(EC2 / "cpu/sysbench.json", {
    "events_per_sec": 12400.5,
    "latency_avg_ms": 0.08,
    "latency_p95_ms": 0.10,
    "total_events": 372015,
    "threads": 192,
    "duration_s": 30,
})
write(EC2 / "cpu/cgroup_throttle.json", {
    "nr_throttled_delta": 0,
    "throttled_usec_delta": 0,
    "before": {"nr_throttled": 0, "throttled_usec": 0},
    "after": {"nr_throttled": 0, "throttled_usec": 0},
})


# ── runpod-pod fixtures ─────────────────────────────────────────────────────────

POD = BASE / "runpod-pod" / "2026-05-06T14:00:00Z"

print("\nGenerating runpod-pod fixtures...")

# manifest -- cgroup quota is set (RunPod restricts CPU)
write(POD / "manifest.json", manifest("runpod-pod", cgroup_cpu_quota="1600000 100000"))

# vLLM serving -- rate 1 (slight degradation)
write(POD / "vllm_serving/rate_1/run_0.json",
      vllm_run(1.0, 127.5, 43.5, 60.0, 8.4, 10.3, 8.2, 9.7))  # warmup
for i, (ttft, tpot) in enumerate([(43.8, 8.4), (43.5, 8.3), (44.0, 8.5), (43.6, 8.4), (43.4, 8.3)], start=1):
    write(POD / f"vllm_serving/rate_1/run_{i}.json",
          vllm_run(1.0, 127.5, ttft, 59.5, tpot, 10.2, 8.2, 9.6))

# vLLM serving -- rate 4
write(POD / "vllm_serving/rate_4/run_0.json",
      vllm_run(3.98, 509.0, 58.0, 84.0, 8.7, 11.3, 8.5, 10.2))  # warmup
for i, (ttft, tpot) in enumerate([(58.2, 8.7), (57.8, 8.6), (58.5, 8.8), (58.0, 8.7), (57.9, 8.6)], start=1):
    write(POD / f"vllm_serving/rate_4/run_{i}.json",
          vllm_run(3.98, 509.0, ttft, 83.0, tpot, 11.1, 8.5, 10.1))

# vLLM serving -- rate inf
write(POD / "vllm_serving/rate_inf/run_0.json",
      vllm_run(18.3, 2342.0, 325.0, 458.0, 9.2, 14.2, 9.0, 13.2))  # warmup
for i, (ttft, tpot) in enumerate([(323.0, 9.2), (327.0, 9.3), (324.0, 9.2), (326.0, 9.3), (325.0, 9.2)], start=1):
    write(POD / f"vllm_serving/rate_inf/run_{i}.json",
          vllm_run(18.3, 2342.0, ttft, 456.0, tpot, 14.0, 9.0, 13.0))

# cold start (slightly slower -- RunPod pod startup overhead)
write(POD / "cold_start/run_0.json", cold_start_run(30.5, 0))  # warmup
for i, dur in enumerate([29.2, 29.5, 28.9, 29.8, 29.3], start=1):
    write(POD / f"cold_start/run_{i}.json", cold_start_run(dur, i))

# fio -- same storage, minimal difference expected
write(POD / "fio/seq_read.json",
      fio_result(bw_bytes=3334668288, iops=3180, lat_p50_ns=302000, lat_p99_ns=1210000, rw="read"))
write(POD / "fio/seq_write.json",
      fio_result(bw_bytes=2087518208, iops=1991, lat_p50_ns=483000, lat_p99_ns=1820000, rw="write"))
write(POD / "fio/rand_read.json",
      fio_result(bw_bytes=1068269568, iops=4075, lat_p50_ns=958000, lat_p99_ns=3230000, rw="read"))
write(POD / "fio/rand_write.json",
      fio_result(bw_bytes=66846720, iops=16320, lat_p50_ns=222000, lat_p99_ns=810000, rw="write"))

# cpu -- cgroup throttling observed in pod
write(POD / "cpu/sysbench.json", {
    "events_per_sec": 12348.2,
    "latency_avg_ms": 0.081,
    "latency_p95_ms": 0.101,
    "total_events": 370446,
    "threads": 192,
    "duration_s": 30,
})
write(POD / "cpu/cgroup_throttle.json", {
    "nr_throttled_delta": 47,
    "throttled_usec_delta": 1280000,
    "before": {"nr_throttled": 1023, "throttled_usec": 45000000},
    "after": {"nr_throttled": 1070, "throttled_usec": 46280000},
})

print("\nDone. All fixtures written.")
