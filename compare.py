#!/usr/bin/env python3
"""compare.py -- Diff two benchmark summaries and output a delta table.

Usage: python compare.py <baseline_results_dir> <treatment_results_dir>

Outputs a table showing baseline vs treatment with percent deltas.
Color-coded: green (<5% delta), yellow (5-15%), red (>15%).
"""

import json
import sys
from pathlib import Path


# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"WARN: Could not load {path}: {e}", file=sys.stderr)
        return None


def color_delta(delta_pct, higher_is_worse=True):
    """Color a delta percentage. For latency, higher is worse. For throughput, lower is worse."""
    if delta_pct is None:
        return "n/a"
    abs_delta = abs(delta_pct)
    # Determine if this delta is bad
    is_bad = (delta_pct > 0 and higher_is_worse) or (delta_pct < 0 and not higher_is_worse)

    if abs_delta < 5:
        color = GREEN
    elif abs_delta < 15:
        color = YELLOW
    else:
        color = RED if is_bad else GREEN

    sign = "+" if delta_pct > 0 else ""
    return f"{color}{sign}{delta_pct:.1f}%{RESET}"


def pct_delta(baseline, treatment):
    """Compute percent delta from baseline to treatment."""
    if baseline is None or treatment is None or baseline == 0:
        return None
    return ((treatment - baseline) / abs(baseline)) * 100


def print_manifest_comparison(m1, m2, name1, name2):
    """Print side-by-side manifest comparison."""
    print(f"\n{BOLD}=== Manifest Comparison ==={RESET}")
    fields = [
        "hostname", "in_container", "gpu_model", "gpu_count", "gpu_driver",
        "cuda_version", "cpu_model", "cpu_count", "memory_gb",
        "vllm_version", "torch_version", "cgroup_cpu_quota", "cgroup_memory_limit",
    ]
    print(f"{'':>22} | {name1:<22} | {name2:<22}")
    print("-" * 72)
    for field in fields:
        v1 = str(m1.get(field, "n/a"))[:22]
        v2 = str(m2.get(field, "n/a"))[:22]
        marker = " *" if v1 != v2 else ""
        print(f"{field:>22} | {v1:<22} | {v2:<22}{marker}")

    # Warnings
    warnings = []
    if m1.get("gpu_model") != m2.get("gpu_model"):
        warnings.append("WARNING: GPU models differ!")
    if m1.get("gpu_count") != m2.get("gpu_count"):
        warnings.append("WARNING: GPU counts differ!")
    if m1.get("vllm_version") != m2.get("vllm_version"):
        warnings.append("WARNING: vLLM versions differ!")
    for w in warnings:
        print(f"\n{RED}{w}{RESET}")


def print_vllm_comparison(s1, s2, name1, name2):
    """Print vLLM serving comparison table."""
    if not s1 or not s2:
        print("\n(vLLM serving data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== vLLM Serving ==={RESET}")

    metrics = [
        ("request_throughput",  "tput req/s",  False),  # higher is better
        ("output_throughput",   "tput tok/s",  False),
        ("median_ttft_ms",      "TTFT p50",    True),   # higher is worse
        ("p99_ttft_ms",         "TTFT p99",    True),
        ("median_tpot_ms",      "TPOT p50",    True),
        ("median_itl_ms",       "ITL p50",     True),
    ]

    print(f"{'Rate':<6} | {'Metric':<12} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 65)

    all_rates = sorted(
        set(list(s1.keys()) + list(s2.keys())),
        key=lambda r: float(r) if r != "inf" else float("inf"),
    )

    for rate in all_rates:
        r1 = s1.get(rate, {})
        r2 = s2.get(rate, {})
        for key, label, higher_is_worse in metrics:
            v1 = r1.get(key, {}).get("median")
            v2 = r2.get(key, {}).get("median")
            delta = pct_delta(v1, v2)
            delta_str = color_delta(delta, higher_is_worse)

            v1_str = f"{v1:.1f}" if v1 is not None else "n/a"
            v2_str = f"{v2:.1f}" if v2 is not None else "n/a"
            print(f"{rate:<6} | {label:<12} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")
        print("-" * 65)


def print_cold_start_comparison(s1, s2, name1, name2):
    """Print cold start comparison."""
    if not s1 or not s2:
        print("\n(Cold start data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== Cold Start ==={RESET}")
    print(f"{'Metric':<8} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 55)

    for key, label in [("median_s", "median"), ("p95_s", "p95")]:
        v1 = s1.get(key)
        v2 = s2.get(key)
        delta = pct_delta(v1, v2)
        delta_str = color_delta(delta, higher_is_worse=True)
        v1_str = f"{v1:.1f}s" if v1 is not None else "n/a"
        v2_str = f"{v2:.1f}s" if v2 is not None else "n/a"
        print(f"{label:<8} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")


def print_fio_comparison(s1, s2, name1, name2):
    """Print fio comparison."""
    if not s1 or not s2:
        print("\n(FIO data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== Storage (fio) ==={RESET}")
    print(f"{'Profile':<12} | {'Metric':<10} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 68)

    for profile in ["seq_read", "seq_write", "rand_read", "rand_write"]:
        p1 = s1.get(profile, {})
        p2 = s2.get(profile, {})
        for key, label, higher_is_worse in [
            ("bw_mbs", "BW MB/s", False),
            ("iops", "IOPS", False),
            ("lat_p50_ms", "Lat p50", True),
            ("lat_p99_ms", "Lat p99", True),
        ]:
            v1 = p1.get(key)
            v2 = p2.get(key)
            delta = pct_delta(v1, v2)
            delta_str = color_delta(delta, higher_is_worse)
            v1_str = f"{v1:.1f}" if v1 is not None else "n/a"
            v2_str = f"{v2:.1f}" if v2 is not None else "n/a"
            print(f"{profile:<12} | {label:<10} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")
        print("-" * 68)


def print_cpu_comparison(s1, s2, name1, name2):
    """Print CPU comparison."""
    if not s1 or not s2:
        print("\n(CPU data missing from one or both runs)")
        return

    print(f"\n{BOLD}=== CPU ==={RESET}")
    print(f"{'Metric':<20} | {name1:>12} | {name2:>12} | {'Delta':>12}")
    print("-" * 62)

    # sysbench
    sb1 = s1.get("sysbench", {})
    sb2 = s2.get("sysbench", {})
    for key, label, higher_is_worse in [
        ("events_per_sec", "events/s", False),
        ("latency_avg_ms", "lat avg ms", True),
        ("latency_p95_ms", "lat p95 ms", True),
    ]:
        v1 = sb1.get(key)
        v2 = sb2.get(key)
        delta = pct_delta(v1, v2)
        delta_str = color_delta(delta, higher_is_worse)
        v1_str = f"{v1:.1f}" if v1 is not None else "n/a"
        v2_str = f"{v2:.1f}" if v2 is not None else "n/a"
        print(f"{label:<20} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")

    # cgroup throttle
    cg1 = s1.get("cgroup_throttle", {})
    cg2 = s2.get("cgroup_throttle", {})
    for key, label in [("nr_throttled_delta", "throttled count"), ("throttled_usec_delta", "throttled usec")]:
        v1 = cg1.get(key)
        v2 = cg2.get(key)
        if v1 is None and v2 is None:
            continue
        v1_str = str(v1) if v1 is not None else "n/a"
        v2_str = str(v2) if v2 is not None else "n/a"
        # Absolute delta for throttle (not percent -- baseline is often 0)
        if v1 is not None and v2 is not None:
            abs_delta = v2 - v1
            if abs_delta == 0:
                delta_str = f"{GREEN}0{RESET}"
            else:
                delta_str = f"{RED}{'+' if abs_delta > 0 else ''}{abs_delta}{RESET}"
        else:
            delta_str = "n/a"
        print(f"{label:<20} | {v1_str:>12} | {v2_str:>12} | {delta_str:>20}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline_dir> <treatment_dir>", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    treat_dir = Path(sys.argv[2])

    # Derive names from directory structure: results/<env-name>/<timestamp>
    base_name = base_dir.parent.name
    treat_name = treat_dir.parent.name

    # Load data
    base_manifest = load_json(base_dir / "manifest.json") or {}
    treat_manifest = load_json(treat_dir / "manifest.json") or {}
    base_summary = load_json(base_dir / "summary.json") or {}
    treat_summary = load_json(treat_dir / "summary.json") or {}

    # Print comparison
    print(f"\n{BOLD}RunPod Benchmark Comparison{RESET}")
    print(f"Baseline:  {base_dir}")
    print(f"Treatment: {treat_dir}")

    print_manifest_comparison(base_manifest, treat_manifest, base_name, treat_name)
    print_vllm_comparison(
        base_summary.get("vllm_serving"),
        treat_summary.get("vllm_serving"),
        base_name, treat_name,
    )
    print_cold_start_comparison(
        base_summary.get("cold_start"),
        treat_summary.get("cold_start"),
        base_name, treat_name,
    )
    print_fio_comparison(
        base_summary.get("fio"),
        treat_summary.get("fio"),
        base_name, treat_name,
    )
    print_cpu_comparison(
        base_summary.get("cpu"),
        treat_summary.get("cpu"),
        base_name, treat_name,
    )
    print()


if __name__ == "__main__":
    main()
