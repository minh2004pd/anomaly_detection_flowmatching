#!/bin/bash
# Run once on a new server to set up the environment
set -e

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$(dirname "$0")"
uv sync
echo "Environment ready. Run: uv run bash train_brats.sh"
