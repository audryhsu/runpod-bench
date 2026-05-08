#!/usr/bin/env bash
# run_all.sh -- Top-level orchestrator for all benchmarks.
#
# Usage:
#   ./run_all.sh --comparison <name> --env-name <ec2-direct|runpod-pod> \
#                [--runs <n>] [--skip-fio] [--skip-cpu] [--skip-cold-start]
#
# Results land in: results/<comparison>/<env-name>/<timestamp>/
#
# Examples:
#   # Bare Docker baseline on EC2 host:
#   ./run_all.sh --comparison 8gpu-32b-tp8 --env-name ec2-direct
#
#   # Inside a Runpod pod:
#   ./run_all.sh --comparison 8gpu-32b-tp8 --env-name runpod-pod
#
# Convention for --comparison: short slug describing the workload, e.g.
#   1gpu-8b-tp1, 8gpu-32b-tp8, h100-70b-tp4
#
# Convention for --env-name: the side of the comparison being run.
#   Recommended values: ec2-direct, runpod-pod (or baseline, treatment).
#
# Legacy mode (no --comparison): writes to results/<env-name>/<timestamp>/.
# Prints a deprecation warning. Kept for back-compat with older docs.

source "$(dirname "$0")/config.sh"

# --- Parse arguments ---
COMPARISON=""
ENV_NAME=""
SKIP_FIO=false
SKIP_CPU=false
SKIP_COLD_START=false

usage() {
  sed -n '4,22p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --comparison)  COMPARISON="$2"; shift 2 ;;
    --env-name)    ENV_NAME="$2"; shift 2 ;;
    --runs)        BENCH_RUNS="$2"; TOTAL_RUNS=$((WARMUP_RUNS + BENCH_RUNS)); shift 2 ;;
    --skip-fio)    SKIP_FIO=true; shift ;;
    --skip-cpu)    SKIP_CPU=true; shift ;;
    --skip-cold-start) SKIP_COLD_START=true; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENV_NAME" ]]; then
  echo "ERROR: --env-name is required (e.g. ec2-direct, runpod-pod)" >&2
  usage >&2
  exit 1
fi

# --- Create results directory ---
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [[ -n "$COMPARISON" ]]; then
  RESULTS_DIR="${SCRIPT_DIR}/results/${COMPARISON}/${ENV_NAME}/${TIMESTAMP}"
else
  echo "WARN: --comparison not provided. Writing to legacy path results/${ENV_NAME}/${TIMESTAMP}/." >&2
  echo "WARN: New runs should use --comparison <slug>, e.g. --comparison 8gpu-32b-tp8." >&2
  RESULTS_DIR="${SCRIPT_DIR}/results/${ENV_NAME}/${TIMESTAMP}"
fi
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "  RunPod Benchmark Harness"
echo "=============================================="
if [[ -n "$COMPARISON" ]]; then
  echo "Comparison:  $COMPARISON"
fi
echo "Environment: $ENV_NAME"
echo "Timestamp:   $TIMESTAMP"
echo "Results:     $RESULTS_DIR"
echo "Runs:        $BENCH_RUNS (+ $WARMUP_RUNS warmup)"
echo "=============================================="
echo ""

# --- Capture manifest ---
echo ">>> Capturing environment manifest..."
"$SCRIPT_DIR/capture_manifest.sh" > "${RESULTS_DIR}/manifest.json"
echo "Manifest saved."
echo ""

# --- Snapshot cgroup state (before vLLM benchmark) ---
read_cgroup_cpu_stat() {
  local stat_file=""
  if [[ -f /sys/fs/cgroup/cpu.stat ]]; then
    stat_file="/sys/fs/cgroup/cpu.stat"
  elif [[ -f /sys/fs/cgroup/cpu/cpu.stat ]]; then
    stat_file="/sys/fs/cgroup/cpu/cpu.stat"
  fi

  if [[ -n "$stat_file" ]]; then
    local nr_throttled
    nr_throttled=$(grep "nr_throttled " "$stat_file" | awk '{print $2}')
    local throttled_usec
    throttled_usec=$(grep "throttled_usec " "$stat_file" | awk '{print $2}')
    # cgroup v1 uses throttled_time (nanoseconds), convert to usec
    if [[ -z "$throttled_usec" ]]; then
      local throttled_time
      throttled_time=$(grep "throttled_time " "$stat_file" | awk '{print $2}')
      if [[ -n "$throttled_time" ]]; then
        throttled_usec=$((throttled_time / 1000))
      fi
    fi
    echo "${nr_throttled:-0} ${throttled_usec:-0}"
  else
    echo "null null"
  fi
}

CGROUP_BEFORE=$(read_cgroup_cpu_stat)
CGROUP_BEFORE_THROTTLED=$(echo "$CGROUP_BEFORE" | awk '{print $1}')
CGROUP_BEFORE_USEC=$(echo "$CGROUP_BEFORE" | awk '{print $2}')

# --- Tier 1: vLLM Serving Sweep ---
echo ">>> Tier 1: vLLM Serving Sweep"
"$SCRIPT_DIR/bench_vllm_serving.sh" "$RESULTS_DIR"
echo ""

# --- Snapshot cgroup state (after vLLM benchmark) ---
CGROUP_AFTER=$(read_cgroup_cpu_stat)
CGROUP_AFTER_THROTTLED=$(echo "$CGROUP_AFTER" | awk '{print $1}')
CGROUP_AFTER_USEC=$(echo "$CGROUP_AFTER" | awk '{print $2}')

# Write cgroup throttle delta
mkdir -p "${RESULTS_DIR}/cpu"
if [[ "$CGROUP_BEFORE_THROTTLED" == "null" ]]; then
  jq -n '{nr_throttled_delta:null, throttled_usec_delta:null, note:"cgroup cpu.stat not available"}' \
    > "${RESULTS_DIR}/cpu/cgroup_throttle.json"
  echo "WARN: cgroup cpu.stat not found -- throttle data unavailable"
else
  NR_DELTA=$((CGROUP_AFTER_THROTTLED - CGROUP_BEFORE_THROTTLED))
  USEC_DELTA=$((CGROUP_AFTER_USEC - CGROUP_BEFORE_USEC))
  jq -n \
    --argjson nr_throttled_delta "$NR_DELTA" \
    --argjson throttled_usec_delta "$USEC_DELTA" \
    --argjson before_nr "$CGROUP_BEFORE_THROTTLED" \
    --argjson after_nr "$CGROUP_AFTER_THROTTLED" \
    --argjson before_usec "$CGROUP_BEFORE_USEC" \
    --argjson after_usec "$CGROUP_AFTER_USEC" \
    '{nr_throttled_delta:$nr_throttled_delta, throttled_usec_delta:$throttled_usec_delta, before:{nr_throttled:$before_nr, throttled_usec:$before_usec}, after:{nr_throttled:$after_nr, throttled_usec:$after_usec}}' \
    > "${RESULTS_DIR}/cpu/cgroup_throttle.json"
  echo "Cgroup throttle delta: nr_throttled=$NR_DELTA, throttled_usec=$USEC_DELTA"
fi
echo ""

# --- Tier 2: Cold Start ---
if [[ "$SKIP_COLD_START" == "false" ]]; then
  echo ">>> Tier 2: Cold Start"
  "$SCRIPT_DIR/bench_cold_start.sh" "$RESULTS_DIR"
  echo ""
else
  echo ">>> Tier 2: Cold Start -- SKIPPED"
  echo ""
fi

# --- Tier 3: Storage (fio) ---
if [[ "$SKIP_FIO" == "false" ]]; then
  echo ">>> Tier 3: Storage (fio)"
  "$SCRIPT_DIR/bench_fio.sh" "$RESULTS_DIR"
  echo ""
else
  echo ">>> Tier 3: Storage (fio) -- SKIPPED"
  echo ""
fi

# --- Tier 4: CPU ---
if [[ "$SKIP_CPU" == "false" ]]; then
  echo ">>> Tier 4: CPU"
  "$SCRIPT_DIR/bench_cpu.sh" "$RESULTS_DIR"
  echo ""
else
  echo ">>> Tier 4: CPU -- SKIPPED"
  echo ""
fi

# --- Summarize ---
echo ">>> Generating summary..."
python "$SCRIPT_DIR/summarize.py" "$RESULTS_DIR"
echo ""

echo "=============================================="
echo "  Benchmark Complete"
echo "=============================================="
echo "Results:  $RESULTS_DIR"
echo "Summary:  ${RESULTS_DIR}/summary.txt"
echo ""

# Suggest the paired environment for comparison
if [[ -n "$COMPARISON" ]]; then
  case "$ENV_NAME" in
    ec2-direct) PAIRED_ENV="runpod-pod" ;;
    runpod-pod) PAIRED_ENV="ec2-direct" ;;
    baseline)   PAIRED_ENV="treatment" ;;
    treatment)  PAIRED_ENV="baseline" ;;
    *)          PAIRED_ENV="" ;;
  esac
  if [[ -n "$PAIRED_ENV" ]]; then
    PAIRED_DIR="${SCRIPT_DIR}/results/${COMPARISON}/${PAIRED_ENV}"
    if [[ -d "$PAIRED_DIR" ]]; then
      LATEST_PAIRED=$(ls -1d "${PAIRED_DIR}"/*/ 2>/dev/null | sort | tail -1 | sed 's:/*$::')
      if [[ -n "$LATEST_PAIRED" ]]; then
        echo "Paired ($PAIRED_ENV) run available at:"
        echo "  $LATEST_PAIRED"
        echo ""
        echo "Compare:"
        echo "  ./compare.py $LATEST_PAIRED $RESULTS_DIR"
        exit 0
      fi
    fi
    echo "No paired $PAIRED_ENV run found yet under results/$COMPARISON/."
    echo "Run the other side, then compare:"
    echo "  ./compare.py results/$COMPARISON/$PAIRED_ENV/<ts> $RESULTS_DIR"
    exit 0
  fi
fi

echo "To compare with another run:"
echo "  ./compare.py $RESULTS_DIR <other_results_dir>"
