#!/bin/sh
set -e

# Port configuration (Railway / Cloud environment compatibility)
PORT="${PORT:-8000}"

# Auto-launch Telegram Bot background worker if token is provided or fallback token is available
BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-8891659055:AAEhcDFlqN192wGJwpqjTBuCsUINe68TQYc}"

if [ -n "${BOT_TOKEN}" ]; then
    echo "[entrypoint] Telegram bot token detected. Starting Telegram Bot worker..."
    export TELEGRAM_BOT_TOKEN="${BOT_TOKEN}"
    python -u -m src.telegram_bot.bot &
fi

# If specific command arguments are passed, execute them
if [ "$#" -gt 0 ]; then
    exec "$@"
else
    # Default production startup: FastAPI server via Uvicorn on dynamic PORT
    echo "[entrypoint] Starting FastAPI server on 0.0.0.0:${PORT}..."
    exec uvicorn src.api.main:app --host 0.0.0.0 --port "$PORT"
fi
