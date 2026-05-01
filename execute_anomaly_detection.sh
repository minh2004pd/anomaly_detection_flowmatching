#!/bin/bash

set -a && source .env 2>/dev/null; set +a

if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./anomaly_results}"

# Auto-find latest checkpoint
CHECKPOINT=""
if [ -f "./output_brats/checkpoint.pth" ]; then
    CHECKPOINT="./output_brats/checkpoint.pth"
else
    # Find latest checkpoint-N.pth
    CHECKPOINT=$(ls -t ./output_brats/checkpoint-*.pth 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ./output_brats/"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

mkdir -p "$OUTPUT_DIR"

uv run python infer_anomaly.py \
  --checkpoint    "$CHECKPOINT" \
  --data_path     "$DATA_PATH" \
  --t             0.8 \
  --step_size     0.02 \
  --cfg_scale     3.0 \
  --num_unhealthy 50 \
  --num_healthy   20 \
  --output_dir    "$OUTPUT_DIR" \
  --device        cuda

echo "Anomaly Detection completed!"

# Send results back to k66
K66_HOST="${K66_HOST:-}"
K66_PORT="${K66_PORT:-22}"
K66_USER="${K66_USER:-k66}"
K66_DIR="${K66_DIR:-/mnt/apple/k66/minhdd/flow-matching-main}"

if [ -n "$K66_HOST" ]; then
    echo "Sending results to k66: ${K66_USER}@${K66_HOST}:${K66_DIR}/anomaly_results ..."
    rsync -avz -e "ssh -p ${K66_PORT} -o StrictHostKeyChecking=no" \
        "$OUTPUT_DIR/" \
        "${K66_USER}@${K66_HOST}:${K66_DIR}/anomaly_results/"
    echo "Done: results saved to ${K66_DIR}/anomaly_results/"
else
    echo "Tip: set K66_HOST in .env to auto-send results to k66."
fi
