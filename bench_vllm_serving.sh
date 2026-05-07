#!/usr/bin/env bash
# bench_vllm_serving.sh -- Sweep request rates with vllm bench serve.
# Usage: ./bench_vllm_serving.sh <results_dir>
# Expects vLLM server to be running already.

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"

# Verify server is reachable
if ! curl -sf "${VLLM_URL}/v1/models" > /dev/null 2>&1; then
  echo "ERROR: vLLM server not reachable at ${VLLM_URL}/v1/models" >&2
  echo "Start it with: ./serve.sh &" >&2
  exit 1
fi

echo "=== vLLM Serving Benchmark ==="
echo "Rates: $REQUEST_RATES"
echo "Runs per rate: $TOTAL_RUNS ($WARMUP_RUNS warmup + $BENCH_RUNS measured)"
echo "Prompts per run: $NUM_PROMPTS"
echo ""

for rate in $REQUEST_RATES; do
  RATE_DIR="${RESULTS_DIR}/vllm_serving/rate_${rate}"
  mkdir -p "$RATE_DIR"

  echo "--- Rate: $rate req/s ---"
  for run in $(seq 0 $((TOTAL_RUNS - 1))); do
    LABEL="run_${run}"
    if [[ $run -lt $WARMUP_RUNS ]]; then
      LABEL="${LABEL} (warmup)"
    fi
    echo "  $LABEL"

    RATE_FLAG="--request-rate $rate"
    if [[ "$rate" == "inf" ]]; then
      RATE_FLAG="--request-rate inf"
    fi

    vllm bench serve \
      --model "$MODEL" \
      --base-url "$VLLM_URL" \
      --endpoint /v1/completions \
      --num-prompts "$NUM_PROMPTS" \
      --random-input-len "$INPUT_LEN" \
      --random-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      $RATE_FLAG \
      --save-result \
      --result-dir "$RATE_DIR" \
      --result-filename "run_${run}.json" \
      2>&1 | tee "${RATE_DIR}/run_${run}.log"

    echo ""
  done
done

echo "=== vLLM Serving Benchmark Complete ==="
echo "Results: ${RESULTS_DIR}/vllm_serving/"
