#!/usr/bin/env bash
# bench_cpu.sh -- sysbench CPU benchmark + cgroup throttle snapshot.
# Usage: ./bench_cpu.sh <results_dir>

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"
CPU_DIR="${RESULTS_DIR}/cpu"
mkdir -p "$CPU_DIR"

THREADS=$(nproc)

echo "=== CPU Benchmark ==="
echo "Threads: $THREADS"
echo "Duration: ${SYSBENCH_DURATION}s"
echo ""

# --- sysbench ---
echo "--- sysbench cpu ---"
SYSBENCH_RAW=$(sysbench cpu --threads="$THREADS" --time="$SYSBENCH_DURATION" run 2>&1)
echo "$SYSBENCH_RAW" > "${CPU_DIR}/sysbench_raw.txt"

# Parse sysbench output
EVENTS_PER_SEC=$(echo "$SYSBENCH_RAW" | grep "events per second:" | awk '{print $NF}')
LAT_AVG=$(echo "$SYSBENCH_RAW" | grep "avg:" | awk '{print $NF}')
LAT_P95=$(echo "$SYSBENCH_RAW" | grep "95th percentile:" | awk '{print $NF}')
TOTAL_EVENTS=$(echo "$SYSBENCH_RAW" | grep "total number of events:" | awk '{print $NF}')

jq -n \
  --argjson events_per_sec "${EVENTS_PER_SEC:-0}" \
  --argjson latency_avg_ms "${LAT_AVG:-0}" \
  --argjson latency_p95_ms "${LAT_P95:-0}" \
  --argjson total_events "${TOTAL_EVENTS:-0}" \
  --argjson threads "$THREADS" \
  --argjson duration "$SYSBENCH_DURATION" \
  '{events_per_sec:$events_per_sec, latency_avg_ms:$latency_avg_ms, latency_p95_ms:$latency_p95_ms, total_events:$total_events, threads:$threads, duration_s:$duration}' \
  > "${CPU_DIR}/sysbench.json"

echo "  Events/sec: $EVENTS_PER_SEC"
echo "  Latency avg: ${LAT_AVG}ms, p95: ${LAT_P95}ms"
echo ""

echo "=== CPU Benchmark Complete ==="
echo "Results: $CPU_DIR/"
