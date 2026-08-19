"""
Telegram Bot service for Diabetes RAG.
Connects Telegram user messages to the FastAPI grounded generation backend.
"""

from __future__ import annotations

import logging
import os
import sys
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("diabetes_rag_bot")

DEFAULT_BOT_TOKEN = "8960337678:AAG11DmdzhD12cjQHZOABDPPX5Kcjg0Z0Iw"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or DEFAULT_BOT_TOKEN
PORT = os.environ.get("PORT", "8000")
API_URL = os.environ.get("API_URL", f"http://127.0.0.1:{PORT}/ask")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if not update.message:
        return
    welcome_text = (
        "🩺 *Diabetes RAG Clinical Assistant*\n\n"
        "Welcome! I provide evidence-based recommendations on Type 2 Diabetes Management "
        "grounded directly in the NICE NG28 clinical guidelines.\n\n"
        "Simply send me a clinical query or question (e.g., 'What is the first-line drug treatment for Type 2 diabetes?')."
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if not update.message:
        return
    help_text = (
        "💡 *How to use Diabetes RAG Bot:*\n\n"
        "• Send any clinical query regarding Type 2 Diabetes management.\n"
        "• All answers are grounded strictly in official NICE guidelines.\n"
        "• Responses include exact clinical recommendations, supporting excerpts, and citations.\n"
        "• If the guidelines do not contain enough evidence, I will refuse rather than hallucinate."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages and query the RAG API backend."""
    if not update.message or not update.message.text:
        return

    user_query = update.message.text.strip()
    logger.info(f"Received query from chat {update.message.chat_id}: {user_query}")

    try:
        # Send typing action indicator
        await update.message.chat.send_action(action="typing")

        port = os.environ.get("PORT", "8000")
        raw_api_url = os.environ.get("API_URL", f"http://127.0.0.1:{port}/ask")
        if ("127.0.0.1:8000" in raw_api_url or "localhost:8000" in raw_api_url) and port != "8000":
            api_url = raw_api_url.replace("127.0.0.1:8000", f"127.0.0.1:{port}").replace("localhost:8000", f"127.0.0.1:{port}")
        else:
            api_url = raw_api_url

        response = None
        last_err = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(
                        api_url,
                        json={"query": user_query, "top_k": 5},
                    )
                if response is not None and response.status_code == 200:
                    break
            except Exception as primary_err:
                last_err = primary_err
                logger.warning(f"Attempt {attempt + 1}/3 calling API ({api_url}) failed: {primary_err}")
                if attempt < 2:
                    import asyncio as aio
                    await aio.sleep(3.0)

        if response is None:
            logger.error(f"Error calling RAG API at {api_url}: {last_err}", exc_info=True)
            await update.message.reply_text(
                "⏳ *RAG Backend Initializing*\n\n"
                "The clinical knowledge base is currently finishing its startup sequence. "
                "Please wait 10–20 seconds and send your question again.",
                parse_mode="Markdown"
            )
            return

        if response is None or response.status_code != 200:
            err_text = response.text if response is not None else "No response"
            status_code = response.status_code if response is not None else 500
            logger.error(f"API returned status {status_code}: {err_text}")
            await update.message.reply_text(
                f"⚠️ Backend API returned error status {status_code}."
            )
            return

        data = response.json()

        # Check refusal
        if data.get("refused"):
            reason = data.get("refusal_reason") or "Insufficient grounded evidence."
            reply_msg = (
                "⚠️ *Query Refused*\n\n"
                "I could not find sufficient grounded evidence in the NICE guidelines to answer your query.\n\n"
                f"*Reason:* `{reason}`"
            )
            await update.message.reply_text(reply_msg, parse_mode="Markdown")
            return

        recommendation = data.get("recommendation", "N/A")
        supporting_excerpt = data.get("supporting_excerpt", "")
        citations = data.get("citations", [])

        citations_formatted = []
        for c in citations:
            doc = c.get("document_name", "NICE Guideline")
            sec = c.get("section_number")
            page = c.get("page_number")
            sec_str = f"Sec {sec}, " if sec else ""
            citations_formatted.append(f"• {doc} ({sec_str}p. {page})")

        citations_str = "\n".join(citations_formatted) if citations_formatted else "• NICE NG28 Guideline"

        formatted_reply = (
            f"📋 *Clinical Recommendation:*\n{recommendation}\n\n"
            f"💬 *Supporting Excerpt:*\n\"{supporting_excerpt}\"\n\n"
            f"📚 *Citations:*\n{citations_str}"
        )

        try:
            await update.message.reply_text(formatted_reply, parse_mode="Markdown")
        except Exception as parse_err:
            logger.warning(f"Markdown parsing error: {parse_err}. Retrying without markdown mode.")
            plain_reply = (
                f"📋 Clinical Recommendation:\n{recommendation}\n\n"
                f"💬 Supporting Excerpt:\n\"{supporting_excerpt}\"\n\n"
                f"📚 Citations:\n{citations_str}"
            )
            await update.message.reply_text(plain_reply)

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await update.message.reply_text(
            f"⚠️ An error occurred while communicating with the RAG API backend: {str(e)}"
        )


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable is not set! Bot worker cannot start.")
        sys.exit(1)

    print(f"=== TELEGRAM BOT WORKER STARTING (Token ending in ...{TELEGRAM_BOT_TOKEN[-6:] if len(TELEGRAM_BOT_TOKEN) >= 6 else '***'}) ===", flush=True)
    print(f"=== Connecting to API endpoint: {API_URL} ===", flush=True)
    logger.info(f"Starting Telegram Bot... Connecting to API at: {API_URL}")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("=== TELEGRAM BOT LONG POLLING STARTED SUCCESSFULLY ===", flush=True)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
