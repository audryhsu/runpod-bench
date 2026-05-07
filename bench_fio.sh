#!/usr/bin/env bash
# bench_fio.sh -- Run 4 fio storage profiles.
# Usage: ./bench_fio.sh <results_dir>

source "$(dirname "$0")/config.sh"

RESULTS_DIR="${1:?Usage: $0 <results_dir>}"
FIO_RESULTS="${RESULTS_DIR}/fio"
mkdir -p "$FIO_RESULTS" "$FIO_DIR"

echo "=== Storage Benchmark (fio) ==="
echo "Test dir: $FIO_DIR"
echo "File size: $FIO_SIZE"
echo "Runtime: ${FIO_RUNTIME}s per profile"
echo ""

run_fio() {
  local name="$1"
  local rw="$2"
  local bs="$3"
  local numjobs="$4"
  local iodepth="$5"
  local description="$6"

  echo "--- $name: $description ---"
  fio \
    --name="$name" \
    --rw="$rw" \
    --bs="$bs" \
    --size="$FIO_SIZE" \
    --numjobs="$numjobs" \
    --iodepth="$iodepth" \
    --direct=1 \
    --ioengine=libaio \
    --directory="$FIO_DIR" \
    --runtime="$FIO_RUNTIME" \
    --time_based \
    --group_reporting \
    --output-format=json \
    --output="${FIO_RESULTS}/${name}.json" \
    2>&1 | tee "${FIO_RESULTS}/${name}.log"

  echo "  -> ${FIO_RESULTS}/${name}.json"
  echo ""
}

run_fio "seq_read"   "read"      "1M"   1 32 "Sequential read (model loading)"
run_fio "seq_write"  "write"     "1M"   1 32 "Sequential write (checkpoint saving)"
run_fio "rand_read"  "randread"  "256k" 4 16 "Random read (dataset loading)"
run_fio "rand_write" "randwrite" "4k"   4 32 "Random write (docker layer extraction)"

# Cleanup test files
rm -f "${FIO_DIR}"/seq_read.* "${FIO_DIR}"/seq_write.* \
      "${FIO_DIR}"/rand_read.* "${FIO_DIR}"/rand_write.*

echo "=== Storage Benchmark Complete ==="
echo "Results: $FIO_RESULTS/"
