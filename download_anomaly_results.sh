#!/bin/bash
# Run on k66 to download anomaly results from HuggingFace
set -a && source .env 2>/dev/null; set +a

HF_REPO="${HF_REPO:-minh2k4/brats-flow-matching}"
HF_TOKEN="${HF_TOKEN:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./anomaly_results}"

mkdir -p "$OUTPUT_DIR"

uv run python -c "
import os
from huggingface_hub import HfApi, hf_hub_download, login, list_repo_files
token = '${HF_TOKEN}' or None
if token:
    login(token=token)
api = HfApi()
files = [f for f in api.list_repo_files('${HF_REPO}', repo_type='dataset') if f.startswith('anomaly_results/')]
print(f'Found {len(files)} files to download...')
for f in files:
    local_path = os.path.join('${OUTPUT_DIR}', os.path.relpath(f, 'anomaly_results'))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    hf_hub_download(repo_id='${HF_REPO}', filename=f, repo_type='dataset', local_dir='.', token=token)
    os.rename(f, local_path)
    print(f'Downloaded: {local_path}')
print('Done!')
"
