#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/root/.venv_brats/bin/python"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
LOG="/tmp/demo.log"

# Kill any existing instance
if pgrep -f "demo.py" > /dev/null 2>&1; then
    echo "[run_demo] Stopping existing demo server…"
    pkill -f "demo.py" || true
    sleep 1
fi

echo "[run_demo] Starting demo on http://${HOST}:${PORT}"
cd "$SCRIPT_DIR"

exec env PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/root/.venv_brats \
    "$PYTHON" -B demo.py --host "$HOST" --port "$PORT" "$@"
