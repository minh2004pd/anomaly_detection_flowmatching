#!/bin/bash
# Monitor training, auto-restart on crash, notify via Telegram.
# Usage: bash monitor_train.sh
# Requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env

set -a && source .env && set +a

LOG_DIR="${LOG_DIR:-./logs}"
LOG="$LOG_DIR/train_brats.log"
MAX_RESTARTS=3
restart_count=0

notify() {
    local msg="[flow-matching] $1"
    echo "$msg"
    if [ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=$(echo "$msg" | head -c 4000)" > /dev/null
    fi
}

mkdir -p "$LOG_DIR"

while [ $restart_count -lt $MAX_RESTARTS ]; do
    notify "Training started (attempt $((restart_count+1))/$MAX_RESTARTS)"

    bash train_brats.sh
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        notify "Training completed successfully!"
        exit 0
    fi

    restart_count=$((restart_count + 1))
    last_error=$(tail -10 "$LOG" 2>/dev/null | tr '\n' ' ')
    notify "Crashed (exit=$exit_code), attempt $restart_count/$MAX_RESTARTS.
Last log: $last_error"

    if [ $restart_count -lt $MAX_RESTARTS ]; then
        notify "Pulling latest code and retrying in 15s..."
        git pull
        sleep 15
    fi
done

notify "Failed after $MAX_RESTARTS attempts. Manual fix needed."
exit 1
