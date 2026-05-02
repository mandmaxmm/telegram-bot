# =========================================================
# 🚀 ENSEMBLE ENGINE V6.2
# - Gestione reale modelli attivi
# - Debug completo
# - Maestro dinamico
# =========================================================

import os
import asyncio
import logging
import threading
import time
from typing import Dict, Tuple, List

import httpx
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

# ───────── CONFIG ─────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("V6.2")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

PORT = int(os.environ.get("PORT") or 10000)

HTTP_CLIENT = httpx.AsyncClient(timeout=60)

# CACHE
CACHE: Dict[str, Tuple[float, str]] = {}
CACHE_TTL = 300

# DEBUG
LAST_DEBUG: Dict[str, any] = {}

# ───────── FLASK ─────────

app_flask = Flask(__name__)

@app_flask.route("/")
def home():
    return {"status": "ok"}, 200

def run_flask():
    app_flask.run(host="0.0.0.0", port=PORT)

# ───────── MODELS ─────────

async def call_groq(prompt):
    if not GROQ_API_KEY:
        return None
    try:
        r = await HTTP_CLIENT.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}]},
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return None

async def call_mistral(prompt):
    if not MISTRAL_API_KEY:
        return None
    try:
        r = await HTTP_CLIENT.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}]},
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return None

async def call_gemini(prompt):
    if not GOOGLE_API_KEY:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
        r = await HTTP_CLIENT.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}]
        })
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return None

# ───────── ENGINE ─────────

async def run_engine(query: str):

    responses: List[Tuple[str, str]] = []

    tasks = await asyncio.gather(
        call_groq(query),
        call_mistral(query),
    )

    if tasks[0]:
        responses.append(("Groq", tasks[0]))
    if tasks[1]:
        responses.append(("Mistral", tasks[1]))

    if not responses:
        return "⚠️ Nessuna risposta disponibile"

    combined = "\n\n".join([r[1] for r in responses])

    gemini_response = await call_gemini(query + "\n\n" + combined)

    # DEBUG INFO
    global LAST_DEBUG
    LAST_DEBUG = {
        "models_used": [r[0] for r in responses],
        "gemini_used": bool(gemini_response)
    }

    if gemini_response:
        return gemini_response

    # fallback intelligente → risposta più lunga
    best = max(responses, key=lambda x: len(x[1]))
    return best[1]

# ───────── COMMANDS ─────────

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = sum(bool(x) for x in [GROQ_API_KEY, MISTRAL_API_KEY])
    msg = f"""
📊 STATUS

Groq: {"✅" if GROQ_API_KEY else "❌"}
Mistral: {"✅" if MISTRAL_API_KEY else "❌"}
Gemini: {"✅" if GOOGLE_API_KEY else "❌ (disattivato)"}
DeepSeek: {"⚠️" if DEEPSEEK_API_KEY else "❌"}

Modelli attivi: {active}
Maestro: {"Gemini" if GOOGLE_API_KEY else "Fallback Expert"}
"""
    await update.message.reply_text(msg)

async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Pool attivo:\n- Groq\n- Mistral\n\nMaster: Gemini (se attivo)"
    )

async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(str(LAST_DEBUG))

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🏓 Pong")

# ───────── TELEGRAM ─────────

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⚙️ Elaborazione...")
    res = await run_engine(update.message.text)
    await msg.delete()
    await update.message.reply_text(res)

# ───────── MAIN ─────────

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    threading.Thread(target=run_flask, daemon=True).start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("debug", debug))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    async def startup():
        await app.bot.delete_webhook(drop_pending_updates=True)

    loop.run_until_complete(startup())

    logger.info("🚀 V6.2 RUNNING")

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
