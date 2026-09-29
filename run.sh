#!/usr/bin/env bash
# Builds, generates data (if missing), runs the pipeline, and saves proof
# of execution to artifacts/.  Usage: ./run.sh [ARCH] [NUM_IMAGES]
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-sm_75}"
COUNT="${2:-200}"

make ARCH="${ARCH}"

if ! python3 -c "import numpy, PIL" 2>/dev/null; then
  pip3 install --user numpy pillow
fi

if [ "$(ls data/input/*.pgm 2>/dev/null | wc -l)" -lt "${COUNT}" ]; then
  python3 scripts/generate_data.py --out data/input --count "${COUNT}"
fi

mkdir -p artifacts data/output
{
  echo "=== $(date -u) ==="
  nvidia-smi || true
  echo "=== Run 1: 4 streams, equalize on ==="
  ./bin/cuda_batch_edge --input_dir data/input --output_dir data/output \
      --streams 4 --log artifacts/timings.csv
  echo "=== Run 2: 1 stream (baseline) ==="
  ./bin/cuda_batch_edge --input_dir data/input --output_dir data/output_1s \
      --streams 1
  echo "=== Output files ==="
  ls data/output | wc -l
} 2>&1 | tee artifacts/execution_log.txt

rm -rf data/output_1s
python3 scripts/make_montage.py || true
