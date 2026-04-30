#!/bin/bash
# Runs on Vast.ai. Monitors training, on crash SSHes to k66 to trigger AI fix.
# Usage: bash monitor_train.sh
# Requires in .env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, K66_HOST, K66_PORT, K66_USER, K66_DIR

set -a && source .env && set +a

LOG_DIR="${LOG_DIR:-./logs}"
LOG="$LOG_DIR/train_brats.log"
MAX_RESTARTS=3
restart_count=0

# k66 connection — set these in .env
K66_HOST="${K66_HOST:-}"
K66_PORT="${K66_PORT:-22}"
K66_USER="${K66_USER:-k66}"
K66_DIR="${K66_DIR:-/mnt/apple/k66/minhdd/flow-matching-main}"

notify() {
    local msg="[flow-matching/vast] $1"
    echo "$msg"
    if [ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=$(echo "$msg" | head -c 4000)" > /dev/null
    fi
}

trigger_ai_fix() {
    local error_log="$1"
    if [ -z "$K66_HOST" ]; then
        notify "K66_HOST not set, skipping AI fix."
        return 1
    fi

    notify "Sending crash log to k66 for AI fix..."
    # Escape the error log and send to k66
    local escaped=$(echo "$error_log" | head -c 3000 | sed "s/'/'\\\\''/g")
    ssh -o StrictHostKeyChecking=no -p "$K66_PORT" "${K66_USER}@${K66_HOST}" \
        "bash ${K66_DIR}/fix_and_notify.sh '$escaped' $(hostname -I | awk '{print $1}') $SSH_PORT" &
    # Don't wait — k66 will SSH back to restart when done
    return 0
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
    last_error=$(tail -30 "$LOG" 2>/dev/null)
    last_error_short=$(echo "$last_error" | tail -5 | tr '\n' ' ')
    notify "Crashed (exit=$exit_code), attempt $restart_count/$MAX_RESTARTS. Last: $last_error_short"

    if [ $restart_count -lt $MAX_RESTARTS ]; then
        if trigger_ai_fix "$last_error"; then
            notify "Waiting 60s for k66 to fix and signal back..."
            sleep 60
            git pull
        else
            notify "Pulling latest and retrying in 15s..."
            git pull
            sleep 15
        fi
    fi
done

notify "Failed after $MAX_RESTARTS attempts. Manual fix needed."
exit 1
