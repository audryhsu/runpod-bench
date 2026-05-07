#!/usr/bin/env python3
"""summarize.py -- Aggregate raw benchmark JSON into summary.json + summary.txt.

Usage: python summarize.py <results_dir>

The results_dir should contain subdirectories: vllm_serving/, cold_start/, fio/, cpu/
"""

import json
import sys
from pathlib import Path
from statistics import median


def percentile(data, p):
    """Compute p-th percentile (0-100) of a sorted list."""
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def load_json(path):
    """Load JSON file, return None if missing or invalid."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARN: Could not load {path}: {e}", file=sys.stderr)
        return None


def summarize_vllm_serving(results_dir, warmup_runs):
    """Summarize vLLM serving benchmark results."""
    serving_dir = results_dir / "vllm_serving"
    if not serving_dir.exists():
        return None

    summary = {}
    for rate_dir in sorted(serving_dir.iterdir()):
        if not rate_dir.is_dir() or not rate_dir.name.startswith("rate_"):
            continue
        rate = rate_dir.name.replace("rate_", "")

        runs = []
        for run_file in sorted(rate_dir.glob("run_*.json")):
            run_num = int(run_file.stem.split("_")[1])
            if run_num < warmup_runs:
                continue  # skip warmup
            data = load_json(run_file)
            if data is not None:
                runs.append(data)

        if not runs:
            continue

        # Extract metrics from vllm bench serve output
        metrics = {}
        for key in [
            "request_throughput",
            "output_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
            "p99_itl_ms",
        ]:
            values = [r[key] for r in runs if key in r]
            if values:
                metrics[key] = {
                    "median": round(median(values), 3),
                    "p95": round(percentile(values, 95), 3),
                    "values": [round(v, 3) for v in values],
                }

        summary[rate] = metrics

    return summary if summary else None


def summarize_cold_start(results_dir, warmup_runs):
    """Summarize cold start benchmark results."""
    cold_dir = results_dir / "cold_start"
    if not cold_dir.exists():
        return None

    durations = []
    for run_file in sorted(cold_dir.glob("run_*.json")):
        run_num = int(run_file.stem.split("_")[1])
        if run_num < warmup_runs:
            continue
        data = load_json(run_file)
        if data and data.get("duration_s") is not None:
            durations.append(data["duration_s"])

    if not durations:
        return None

    return {
        "median_s": round(median(durations), 2),
        "p95_s": round(percentile(durations, 95), 2),
        "values": [round(d, 2) for d in durations],
    }


def summarize_fio(results_dir):
    """Summarize fio benchmark results."""
    fio_dir = results_dir / "fio"
    if not fio_dir.exists():
        return None

    summary = {}
    for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
        data = load_json(fio_dir / f"{profile}.json")
        if data is None:
            continue

        jobs = data.get("jobs", [{}])
        if not jobs:
            continue
        job = jobs[0]

        # Determine read or write based on profile
        if "read" in profile:
            io = job.get("read", {})
        else:
            io = job.get("write", {})

        bw_bytes = io.get("bw_bytes", io.get("bw", 0) * 1024)
        bw_mbs = round(bw_bytes / (1024 * 1024), 1)
        iops = round(io.get("iops", 0), 1)
        lat_ns = io.get("clat_ns", io.get("lat_ns", {}))
        lat_p50_ms = round(lat_ns.get("percentile", {}).get("50.000000", 0) / 1e6, 3)
        lat_p99_ms = round(lat_ns.get("percentile", {}).get("99.000000", 0) / 1e6, 3)

        summary[profile] = {
            "bw_mbs": bw_mbs,
            "iops": iops,
            "lat_p50_ms": lat_p50_ms,
            "lat_p99_ms": lat_p99_ms,
        }

    return summary if summary else None


def summarize_cpu(results_dir):
    """Summarize CPU benchmark results."""
    cpu_dir = results_dir / "cpu"
    if not cpu_dir.exists():
        return None

    summary = {}

    sysbench = load_json(cpu_dir / "sysbench.json")
    if sysbench:
        summary["sysbench"] = sysbench

    cgroup = load_json(cpu_dir / "cgroup_throttle.json")
    if cgroup:
        summary["cgroup_throttle"] = cgroup

    return summary if summary else None


def format_summary_txt(summary):
    """Format summary as human-readable text."""
    lines = []

    # vLLM Serving
    vllm = summary.get("vllm_serving")
    if vllm:
        lines.append("=== vLLM Serving ===")
        lines.append(
            f"{'Rate':<6} | {'Tput req/s':>10} | {'Tput tok/s':>10} | {'TTFT p50':>10} | {'TTFT p99':>10} | {'TPOT p50':>10}"
        )
        lines.append("-" * 75)
        for rate in sorted(vllm.keys(), key=lambda r: float(r) if r != "inf" else float("inf")):
            m = vllm[rate]
            req_s = m.get("request_throughput", {}).get("median", "-")
            tok_s = m.get("output_throughput", {}).get("median", "-")
            ttft_p50 = m.get("median_ttft_ms", {}).get("median", "-")
            ttft_p99 = m.get("p99_ttft_ms", {}).get("median", "-")
            tpot_p50 = m.get("median_tpot_ms", {}).get("median", "-")

            def fmt(v):
                return f"{v}" if v == "-" else f"{v:.1f}"

            lines.append(
                f"{rate:<6} | {fmt(req_s):>10} | {fmt(tok_s):>10} | {fmt(ttft_p50):>10} | {fmt(ttft_p99):>10} | {fmt(tpot_p50):>10}"
            )
        lines.append("")

    # Cold Start
    cold = summary.get("cold_start")
    if cold:
        lines.append("=== Cold Start ===")
        lines.append(f"Median: {cold['median_s']}s | p95: {cold['p95_s']}s")
        lines.append("")

    # FIO
    fio = summary.get("fio")
    if fio:
        lines.append("=== Storage (fio) ===")
        lines.append(
            f"{'Profile':<12} | {'BW (MB/s)':>10} | {'IOPS':>10} | {'Lat p50':>10} | {'Lat p99':>10}"
        )
        lines.append("-" * 60)
        for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
            if profile in fio:
                m = fio[profile]
                lines.append(
                    f"{profile:<12} | {m['bw_mbs']:>10.1f} | {m['iops']:>10.1f} | {m['lat_p50_ms']:>8.3f}ms | {m['lat_p99_ms']:>8.3f}ms"
                )
        lines.append("")

    # CPU
    cpu = summary.get("cpu")
    if cpu:
        lines.append("=== CPU ===")
        sb = cpu.get("sysbench", {})
        if sb:
            lines.append(
                f"sysbench: {sb.get('events_per_sec', '-')} events/s, "
                f"avg lat {sb.get('latency_avg_ms', '-')}ms, "
                f"p95 lat {sb.get('latency_p95_ms', '-')}ms"
            )
        cg = cpu.get("cgroup_throttle", {})
        if cg:
            nr = cg.get("nr_throttled_delta", "n/a")
            usec = cg.get("throttled_usec_delta", "n/a")
            lines.append(f"cgroup throttle: nr_throttled={nr}, throttled_usec={usec}")
        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load config from manifest
    manifest = load_json(results_dir / "manifest.json")
    warmup_runs = 1
    if manifest and "config_snapshot" in manifest:
        warmup_runs = int(manifest["config_snapshot"].get("warmup_runs", 1))

    summary = {}

    print("Summarizing vLLM serving...")
    vllm = summarize_vllm_serving(results_dir, warmup_runs)
    if vllm:
        summary["vllm_serving"] = vllm

    print("Summarizing cold start...")
    cold = summarize_cold_start(results_dir, warmup_runs)
    if cold:
        summary["cold_start"] = cold

    print("Summarizing fio...")
    fio = summarize_fio(results_dir)
    if fio:
        summary["fio"] = fio

    print("Summarizing CPU...")
    cpu = summarize_cpu(results_dir)
    if cpu:
        summary["cpu"] = cpu

    # Write summary.json
    summary_json_path = results_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_json_path}")

    # Write summary.txt
    txt = format_summary_txt(summary)
    summary_txt_path = results_dir / "summary.txt"
    with open(summary_txt_path, "w") as f:
        f.write(txt)
    print(f"Wrote {summary_txt_path}")
    print()
    print(txt)


if __name__ == "__main__":
    main()
