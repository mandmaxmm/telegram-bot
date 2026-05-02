import os
import asyncio
import logging
import threading
import signal
import time
from typing import Optional, Callable, Awaitable, Dict, Tuple

import httpx
from flask import Flask, request, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnsembleV5")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

PORT = int(os.environ.get("PORT", 10000))

# API KEYS
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# HTTP CLIENT PERSISTENTE
HTTP_CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)

# CACHE (TTL)
CACHE: Dict[str, Tuple[float, str]] = {}
CACHE_TTL = 300  # 5 minuti

# RANKING BASE (dinamico)
MODEL_SCORES = {
    "groq": 1.0,
    "deepseek": 1.2,
    "mistral": 1.1,
}

# ─────────────────────────────────────────────
# FLASK SERVER
# ─────────────────────────────────────────────

flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return {"status": "ok"}, 200

@flask_app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(run_ensemble_engine(query))
        return jsonify({"response": result})
    except Exception:
        return jsonify({"error": "internal"}), 500

def run_flask():
    flask_app.run(host="0.0.0.0", port=PORT)

# ─────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────

def get_cache(key: str) -> Optional[str]:
    if key in CACHE:
        ts, val = CACHE[key]
        if time.time() - ts < CACHE_TTL:
            return val
    return None

def set_cache(key: str, value: str):
    CACHE[key] = (time.time(), value)

# ─────────────────────────────────────────────
# EXPERT CALLS
# ─────────────────────────────────────────────

async def call_groq(prompt: str):
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

async def call_deepseek(prompt: str):
    if not DEEPSEEK_API_KEY:
        return None
    try:
        r = await HTTP_CLIENT.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]},
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return None

async def call_mistral(prompt: str):
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

# ─────────────────────────────────────────────
# EXPERT GATHER + RANKING
# ─────────────────────────────────────────────

def score_response(text: str, model: str) -> float:
    length_score = min(len(text) / 500, 1)
    base = MODEL_SCORES.get(model, 1)
    return length_score * base

async def gather_experts(prompt: str):
    tasks = [
        call_groq(prompt),
        call_deepseek(prompt),
        call_mistral(prompt),
    ]

    results = await asyncio.gather(*tasks)

    labeled = {
        "groq": results[0],
        "deepseek": results[1],
        "mistral": results[2],
    }

    valid = {k: v for k, v in labeled.items() if v}

    ranked = sorted(
        valid.items(),
        key=lambda x: score_response(x[1], x[0]),
        reverse=True
    )

    return ranked

# ─────────────────────────────────────────────
# MASTER (SEMPLIFICATO + ROBUSTO)
# ─────────────────────────────────────────────

async def synthesize(prompt: str, ranked):
    if GOOGLE_API_KEY:
        try:
            r = await HTTP_CLIENT.post(
                f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except:
            pass

    # fallback → miglior expert
    return ranked[0][1]

# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────

async def run_ensemble_engine(query: str):

    cached = get_cache(query)
    if cached:
        return cached

    try:
        ranked = await asyncio.wait_for(gather_experts(query), timeout=40)
    except asyncio.TimeoutError:
        return "⚠️ Timeout sistema"

    if not ranked:
        return "⚠️ Nessuna risposta disponibile"

    combined = "\n\n".join([r[1] for r in ranked])

    final = await synthesize(query + "\n\n" + combined, ranked)

    set_cache(query, final)

    return final

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⚙️ Elaborazione...")

    try:
        res = await run_ensemble_engine(update.message.text)
    except Exception:
        res = "Errore interno"

    await msg.delete()
    await update.message.reply_text(res)

# ─────────────────────────────────────────────
# SHUTDOWN
# ─────────────────────────────────────────────

def setup_shutdown(app):
    async def shutdown():
        await app.stop()
        await app.shutdown()

    signal.signal(signal.SIGTERM, lambda s, f: asyncio.create_task(shutdown()))
    signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(shutdown()))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    threading.Thread(target=run_flask, daemon=True).start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    setup_shutdown(app)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
