#!/bin/bash

if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
fi

uv run python infer_anomaly.py \
  --checkpoint    ./output_brats/checkpoint.pth \
  --data_path     "$DATA_PATH" \
  --t             0.8 \
  --step_size     0.02 \
  --cfg_scale     3.0 \
  --num_unhealthy 50 \
  --num_healthy   20 \
  --output_dir    anomaly_results \
  --device        cuda

echo "Anomaly Detection completed!"
