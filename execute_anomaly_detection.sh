#!/bin/bash

set -a && source .env 2>/dev/null; set +a

# --update : enable CFG-Zero* (optimized scale + zero-init) at inference.
#            Plug-in only — no retraining needed.
USE_CFG_ZERO_STAR=0
ZERO_INIT_STEPS="${ZERO_INIT_STEPS:-1}"
for arg in "$@"; do
    case "$arg" in
        --update) USE_CFG_ZERO_STAR=1 ;;
    esac
done

if [ -d "/workspace/brats2021" ]; then
    DATA_PATH="${DATA_PATH:-/workspace/brats2021}"
else
    DATA_PATH="${DATA_PATH:-$(cd "$(dirname "$0")/.." && pwd)/data/brats2021}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./anomaly_results}"
HF_REPO="${HF_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Auto-find latest checkpoint
CHECKPOINT=""
if [ -f "./output_brats/checkpoint.pth" ]; then
    CHECKPOINT="./output_brats/checkpoint.pth"
else
    CHECKPOINT=$(ls -t ./output_brats/checkpoint-*.pth 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in ./output_brats/"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [ "$USE_CFG_ZERO_STAR" -eq 1 ]; then
    EXTRA_ARGS+=(--cfg_zero_star --zero_init_steps "$ZERO_INIT_STEPS")
    echo "CFG-Zero* enabled (zero_init_steps=$ZERO_INIT_STEPS)"
fi

uv run python infer_anomaly.py \
  --checkpoint    "$CHECKPOINT" \
  --data_path     "$DATA_PATH" \
  --t             0.8 \
  --step_size     0.02 \
  --cfg_scale     3.0 \
  --num_unhealthy 20 \
  --num_healthy   20 \
  --output_dir    "$OUTPUT_DIR" \
  --device        cuda \
  "${EXTRA_ARGS[@]}"

echo "Anomaly Detection completed!"

# Upload results to HuggingFace so k66 can download
if [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
    echo "Uploading anomaly results to HuggingFace: $HF_REPO ..."
    uv run python -c "
import os, glob
from huggingface_hub import HfApi, login
login(token='${HF_TOKEN}')
api = HfApi()
api.create_repo('${HF_REPO}', repo_type='dataset', exist_ok=True, private=True)
for f in sorted(glob.glob('${OUTPUT_DIR}/**/*', recursive=True)):
    if os.path.isfile(f):
        repo_path = 'anomaly_results/' + os.path.relpath(f, '${OUTPUT_DIR}')
        api.upload_file(path_or_fileobj=f, path_in_repo=repo_path, repo_id='${HF_REPO}', repo_type='dataset')
        print(f'Uploaded: {f}')
print('Done: https://huggingface.co/datasets/${HF_REPO}')
"
else
    echo "Tip: set HF_REPO and HF_TOKEN in .env to auto-upload results."
fi
