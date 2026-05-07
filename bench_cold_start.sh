#!/usr/bin/env bash
# bench_cold_start.sh -- Measure vLLM cold start time (model load to ready).
# Usage: ./bench_cold_start.sh <results_dir>
# Will kill and restart the vLLM server for each run.

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"
COLD_DIR="${RESULTS_DIR}/cold_start"
mkdir -p "$COLD_DIR"

echo "=== Cold Start Benchmark ==="
echo "Runs: $TOTAL_RUNS ($WARMUP_RUNS warmup + $BENCH_RUNS measured)"
echo ""

kill_vllm() {
  # Kill any vllm serve processes
  pkill -f "vllm serve" 2>/dev/null || true
  # Wait for process to fully exit
  for i in $(seq 1 30); do
    if ! pgrep -f "vllm serve" > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  # Force kill if still running
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 2
}

wait_for_ready() {
  local max_wait=600  # 10 minutes max
  local elapsed=0
  while [[ $elapsed -lt $max_wait ]]; do
    if curl -sf "${VLLM_URL}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "ERROR: vLLM server did not become ready within ${max_wait}s" >&2
  return 1
}

for run in $(seq 0 $((TOTAL_RUNS - 1))); do
  LABEL="run_${run}"
  if [[ $run -lt $WARMUP_RUNS ]]; then
    LABEL="${LABEL} (warmup)"
  fi
  echo "--- $LABEL ---"

  # Kill existing server
  echo "  Stopping vLLM server..."
  kill_vllm

  # Drop page cache (best-effort)
  if sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
    echo "  Page cache dropped"
  else
    echo "  WARN: Could not drop page cache (needs root). Continuing."
  fi

  # Record start time and launch server
  START_EPOCH=$(date +%s.%N)
  echo "  Starting vLLM server..."
  "$SCRIPT_DIR/serve.sh" > "${COLD_DIR}/run_${run}_server.log" 2>&1 &
  SERVER_PID=$!

  # Wait for ready
  if wait_for_ready; then
    READY_EPOCH=$(date +%s.%N)
    DURATION=$(echo "$READY_EPOCH - $START_EPOCH" | bc)
    echo "  Ready in ${DURATION}s"

    jq -n \
      --argjson start_epoch "$START_EPOCH" \
      --argjson ready_epoch "$READY_EPOCH" \
      --argjson duration_s "$DURATION" \
      --argjson run "$run" \
      '{start_epoch:$start_epoch, ready_epoch:$ready_epoch, duration_s:$duration_s, run:$run}' \
      > "${COLD_DIR}/run_${run}.json"
  else
    echo "  FAIL: Server did not start"
    jq -n \
      --argjson start_epoch "$START_EPOCH" \
      --argjson run "$run" \
      '{start_epoch:$start_epoch, ready_epoch:null, duration_s:null, run:$run, error:"timeout"}' \
      > "${COLD_DIR}/run_${run}.json"
  fi

  echo ""
done

# Leave server running for subsequent benchmarks
echo "=== Cold Start Benchmark Complete ==="
echo "Server left running (PID: $SERVER_PID)"
echo "Results: $COLD_DIR/"
