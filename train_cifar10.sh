#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate flow_matching

# Train CIFAR-10 with EMA
python train.py \
  --dataset=cifar10 \
  --batch_size=64 \
  --accum_iter=1 \
  --eval_frequency=5 \
  --epochs=20 \
  --class_drop_prob=1.0 \
  --cfg_scale=0.0 \
  --use_ema \
  --data_path=./data \
  --fid_samples=10000 \
  --output_dir=./output_cifar10 > logs/training_logs.log

echo "Training completed!"
