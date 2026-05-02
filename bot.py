# =========================================================
# 🚀 ENSEMBLE ENGINE V6
# Fix:
# - Telegram 409 Conflict definitivo
# - Python 3.14 asyncio fix
# - Gemini endpoint fix
# - Fault tolerance migliorata
# =========================================================

import os
import asyncio
import logging
import threading
import signal
import time
from typing import Dict, Tuple

import httpx
from flask import Flask, request, jsonify
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ───────────────── CONFIG ─────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleV6")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

def get_port():
    try:
        return int(os.environ.get("PORT") or 10000)
    except:
        return 10000

PORT = get_port()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# HTTP CLIENT
HTTP_CLIENT = httpx.AsyncClient(timeout=60)

# CACHE
CACHE: Dict[str, Tuple[float, str]] = {}
CACHE_TTL = 300

# ───────────────── FLASK ─────────────────

app_flask = Flask(__name__)

@app_flask.route("/")
def health():
    return {"status": "ok"}, 200

@app_flask.route("/api/ask", methods=["POST"])
def api():
    data = request.json
    query = data.get("query")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    result = loop.run_until_complete(run_engine(query))
    return jsonify({"response": result})

def run_flask():
    app_flask.run(host="0.0.0.0", port=PORT)

# ───────────────── CACHE ─────────────────

def get_cache(q):
    if q in CACHE:
        ts, val = CACHE[q]
        if time.time() - ts < CACHE_TTL:
            return val
    return None

def set_cache(q, val):
    CACHE[q] = (time.time(), val)

# ───────────────── MODELS ─────────────────

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

# ───────────────── MASTER ─────────────────

async def call_gemini(prompt):
    if not GOOGLE_API_KEY:
        return None
    try:
        r = await HTTP_CLIENT.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return None

# ───────────────── ENGINE ─────────────────

async def run_engine(query):

    cached = get_cache(query)
    if cached:
        return cached

    tasks = await asyncio.gather(
        call_groq(query),
        call_mistral(query),
    )

    responses = [r for r in tasks if r]

    if not responses:
        return "⚠️ Nessuna risposta"

    combined = "\n\n".join(responses)

    final = await call_gemini(query + "\n\n" + combined)

    if not final:
        final = responses[0]

    set_cache(query, final)

    return final

# ───────────────── TELEGRAM ─────────────────

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⚙️ Elaborazione...")
    res = await run_engine(update.message.text)
    await msg.delete()
    await update.message.reply_text(res)

# ───────────────── MAIN ─────────────────

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    threading.Thread(target=run_flask, daemon=True).start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))

    async def startup():
        await app.bot.delete_webhook(drop_pending_updates=True)

    loop.run_until_complete(startup())

    logger.info("🚀 V6 RUNNING")

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
