#!/usr/bin/env bash
# serve.sh -- Launch vLLM server with consistent flags.
# Blocks until the server is ready, then stays in foreground.
# Run in background: ./serve.sh &

source "$(dirname "$0")/config.sh"

ENFORCE_EAGER_FLAG=""
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  ENFORCE_EAGER_FLAG="--enforce-eager"
fi

echo "=== Starting vLLM server ==="
echo "Model: $MODEL"
echo "Port: $VLLM_PORT"
echo "GPU memory utilization: $VLLM_GPU_MEM_UTIL"
echo "Max model len: $VLLM_MAX_MODEL_LEN"
echo "Tensor parallel size: $VLLM_TP_SIZE"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --dtype "$VLLM_DTYPE" \
  $ENFORCE_EAGER_FLAG \
  --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --tensor-parallel-size "$VLLM_TP_SIZE" \
  --seed "$SEED"
