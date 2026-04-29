#!/bin/bash

PYTHON=/home/k66/miniconda3/envs/flow_matching/bin/python
DATA_PATH=/mnt/apple/k66/minhdd/data/brats2021

$PYTHON infer_anomaly.py \
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
