#!/bin/bash

DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_brats}"
LOG_DIR="${LOG_DIR:-./logs}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Train BraTS (healthy + unhealthy, 80-20 split built into BraTSPreprocessedDataset)
# use_preprocessed loads from .npy slices with auto 80-20 case-level split (seed=42)
# Gradient checkpointing is enabled via model_configs.py ("brats" config has use_checkpoint=True)
uv run python train.py \
  --dataset=brats \
  --data_path="$DATA_PATH" \
  --use_preprocessed \
  --image_size=256 \
  --batch_size=4 \
  --accum_iter=8 \
  --epochs=50 \
  --lr=1e-4 \
  --lr_scheduler=cosine \
  --min_lr=1e-6 \
  --precision=bf16 \
  --class_drop_prob=0.15 \
  --cfg_scale=3.0 \
  --use_ema \
  --eval_frequency=10 \
  --fid_samples=50 \
  --num_workers=6 \
  --output_dir="$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/train_brats.log"

echo "Training completed!"

# Upload checkpoint to HuggingFace if HF_REPO is set
# Usage: HF_REPO=vipghn2003/brats-flow-matching HF_TOKEN=hf_xxx bash train_brats.sh
if [ -n "$HF_REPO" ]; then
    echo "Uploading checkpoint to HuggingFace: $HF_REPO ..."
    uv run python hf_upload.py \
        --repo "$HF_REPO" \
        --checkpoint "$OUTPUT_DIR/checkpoint.pth" \
        --output_dir "$OUTPUT_DIR" \
        --all \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
else
    echo "Tip: set HF_REPO=username/repo-name to auto-upload checkpoint after training."
fi