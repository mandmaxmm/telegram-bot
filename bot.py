import os
import asyncio
import logging
import threading
from typing import List, Dict, Any

# Librerie Core
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Caricamento variabili d'ambiente
load_dotenv()

# Configurazione Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- SEZIONE 1: SERVER FLASK (Fix per Render Port Scan) ---
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Ensemble Engine V4.0 Active", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# --- SEZIONE 2: MOTORE DI INTELLIGENZA (API Calls) ---
class EnsembleEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(25.0, connect=5.0)
        self.headers = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "grok": os.getenv("XAI_API_KEY") 
        }

    async def call_openai_compatible(self, client: httpx.AsyncClient, url: str, key: str, model: str, prompt: str) -> str:
        if not key: return ""
        try:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
            response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Errore su {model}: {str(e)}")
            return ""

    async def call_gemini(self, client: httpx.AsyncClient, prompt: str) -> str:
        if not self.headers['gemini']: return ""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.headers['gemini']}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = await client.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"Errore su Gemini: {str(e)}")
            return ""

    async def get_ensemble_response(self, user_query: str) -> str:
        async with httpx.AsyncClient() as client:
            tasks = [
                self.call_openai_compatible(client, "https://api.groq.com/openai/v1/chat/completions", self.headers['groq'], "llama-3.3-70b-versatile", user_query),
                self.call_openai_compatible(client, "https://api.deepseek.com/chat/completions", self.headers['deepseek'], "deepseek-chat", user_query),
                self.call_openai_compatible(client, "https://api.mistral.ai/v1/chat/completions", self.headers['mistral'], "mistral-large-latest", user_query),
                self.call_openai_compatible(client, "https://api.x.ai/v1/chat/completions", self.headers['grok'], "grok-beta", user_query)
            ]
            responses = await asyncio.gather(*tasks)
            valid_responses = [r for r in responses if r]
            if not valid_responses: return "Tutti i modelli esperti sono offline."

            context_for_master = "\n\n".join([f"Esperto {i+1}: {resp}" for i, resp in enumerate(valid_responses)])
            synthesis_prompt = f"Sintetizza in una risposta finale autoritaria: '{user_query}'\n\nContributi:\n{context_for_master}"
            
            final_output = await self.call_gemini(client, synthesis_prompt)
            if not final_output:
                final_output = await self.call_openai_compatible(client, "https://api.deepseek.com/chat/completions", self.headers['deepseek'], "deepseek-chat", synthesis_prompt)
            return final_output or "Errore di sintesi."

# --- SEZIONE 3: HANDLERS ---
engine = EnsembleEngine()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Ensemble V4.0 Attivo.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("🤖 Elaborazione...")
    response = await engine.get_ensemble_response(update.message.text)
    await status_msg.edit_text(response, parse_mode='Markdown')

# --- SEZIONE 4: AVVIO ASINCRONO (Fix per Python 3.14) ---
async def main():
    # Avvio Flask
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info("Server Flask avviato.")

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN mancante!")
        return

    # Configurazione Application
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Gestione manuale del ciclo di vita per evitare RuntimeError
    async with application:
        await application.initialize()
        await application.start()
        logger.info("Polling Telegram avviato correttamente.")
        await application.updater.start_polling(drop_pending_updates=True)
        
        # Mantiene il bot in esecuzione finché non viene fermato
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        # Questo è il punto critico: creiamo esplicitamente il loop
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot fermato dall'utente.")
