#!/bin/bash
# Run once on a new server to set up the environment
set -e

DATA_PATH="${DATA_PATH:-./data/brats2021}"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")"
uv sync

# Generate BraTS split files if not present
if [ ! -f "$DATA_PATH/preprocessed_split.json" ]; then
    echo "Generating BraTS split files..."
    uv run python create_brats_split.py --data_path "$DATA_PATH"
else
    echo "Split files already exist, skipping."
fi

echo "Environment ready. Run: DATA_PATH=$DATA_PATH bash train_brats.sh"
