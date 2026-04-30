#!/bin/bash
# Runs on k66. Called by monitor_train.sh on Vast.ai via SSH.
# Usage: bash fix_and_notify.sh "<error_log>" "<vast_host>" "<vast_port>"

set -a && source "$(dirname "$0")/.env" && set +a

ERROR_LOG="$1"
VAST_HOST="${2:-27.65.63.200}"
VAST_PORT="${3:-55041}"
VAST_DIR="${VAST_DIR:-/workspace/anomaly_detection_flowmatching}"
REPO_DIR="$(dirname "$0")"

notify() {
    local msg="[flow-matching/k66] $1"
    echo "$msg"
    if [ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=$(echo "$msg" | head -c 4000)" > /dev/null
    fi
}

notify "Received crash log. Asking Claude to fix..."

cd "$REPO_DIR"

# Call Claude CLI to analyze and fix
claude --dangerously-skip-permissions -p "
You are fixing a crash in a PyTorch training script.
Repository is at: $REPO_DIR

Here is the tail of the training log showing the crash:
<error_log>
$ERROR_LOG
</error_log>

Tasks:
1. Read the relevant source files to understand the error.
2. Fix the bug.
3. Run: git add -A && git commit -m 'fix: <short description>' && git push
Do NOT ask questions. Fix and push directly.
" 2>&1 | tail -30

if [ $? -eq 0 ]; then
    notify "Claude pushed a fix. Signaling Vast.ai to pull and restart..."
    ssh -o StrictHostKeyChecking=no -p "$VAST_PORT" root@"$VAST_HOST" \
        "cd $VAST_DIR && git pull && set -a && source .env && set +a && bash monitor_train.sh > logs/train_brats.log 2>&1 &"
    notify "Vast.ai restarted training."
else
    notify "Claude fix failed. Manual intervention needed."
fi
