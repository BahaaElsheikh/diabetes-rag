#!/bin/sh
set -e

# Port configuration (Railway / Cloud environment compatibility)
PORT="${PORT:-8000}"

# Optional combined mode: launch Telegram bot worker alongside API if requested
if [ "${START_BOT}" = "true" ] || [ "${ENABLE_TELEGRAM_BOT}" = "true" ]; then
    if [ -n "${TELEGRAM_BOT_TOKEN}" ]; then
        echo "[entrypoint] Starting Telegram Bot background worker..."
        python -u -m src.telegram_bot.bot &
    else
        echo "[entrypoint] WARNING: START_BOT is enabled but TELEGRAM_BOT_TOKEN is not set. Skipping bot startup."
    fi
fi

# If specific command arguments are passed, execute them
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Default production startup: FastAPI server via Uvicorn
    echo "[entrypoint] Starting FastAPI server on 0.0.0.0:${PORT}..."
    exec uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT"
fi
