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
    # Flask gira su thread separato per non bloccare il loop asincrono del bot
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
            "grok": os.getenv("GROK_API_KEY")
        }

    async def call_openai_compatible(self, client: httpx.AsyncClient, url: str, key: str, model: str, prompt: str) -> str:
        """Helper per modelli con standard OpenAI (Groq, DeepSeek, Mistral, xAI)"""
        try:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            response = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Errore su {model}: {str(e)}")
            return ""

    async def call_gemini(self, client: httpx.AsyncClient, prompt: str) -> str:
        """Chiamata specifica per Google Gemini"""
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
        """Orchestrazione Parallela degli Esperti e Sintesi"""
        async with httpx.AsyncClient() as client:
            # 1. Interrogazione Parallela del Pool Esperti
            tasks = [
                self.call_openai_compatible(client, "https://api.groq.com/openai/v1/chat/completions", self.headers['groq'], "llama-3.3-70b-versatile", user_query),
                self.call_openai_compatible(client, "https://api.deepseek.com/chat/completions", self.headers['deepseek'], "deepseek-chat", user_query),
                self.call_openai_compatible(client, "https://api.mistral.ai/v1/chat/completions", self.headers['mistral'], "mistral-large-latest", user_query),
                self.call_openai_compatible(client, "https://api.x.ai/v1/chat/completions", self.headers['grok'], "grok-beta", user_query)
            ]
            
            responses = await asyncio.gather(*tasks)
            # Filtro risposte vuote (Fault Tolerance)
            valid_responses = [r for r in responses if r]
            
            if not valid_responses:
                return "Mi dispiace, ma tutti i modelli esperti sono attualmente non raggiungibili."

            # 2. Fase di Sintesi (Modello Maestro)
            context_for_master = "\n\n".join([f"Esperto {i+1}: {resp}" for i, resp in enumerate(valid_responses)])
            synthesis_prompt = (
                f"Agisci come Modello Maestro. Analizza le seguenti risposte fornite da esperti diversi alla domanda dell'utente: '{user_query}'\n\n"
                f"{context_for_master}\n\n"
                "Genera una risposta finale autoritaria, coerente e strutturata che sintetizzi il meglio di ogni contributo."
            )

            final_output = await self.call_gemini(client, synthesis_prompt)

            # 3. Fallback di Sintesi (Se Gemini fallisce)
            if not final_output:
                logger.warning("Gemini fallito, attivo Fallback su DeepSeek")
                final_output = await self.call_openai_compatible(
                    client, "https://api.deepseek.com/chat/completions", 
                    self.headers['deepseek'], "deepseek-chat", synthesis_prompt
                )

            return final_output or "Errore critico nella fase di sintesi."

# --- SEZIONE 3: LOGICA TELEGRAM HANDLER ---

engine = EnsembleEngine()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Ensemble Engine V4.0 Online. Invia una domanda per interrogare il pool di esperti.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Feedback visivo immediato
    status_msg = await update.message.reply_text("🤖 Interrogando il pool di esperti...")
    
    try:
        response = await engine.get_ensemble_response(user_text)
        await status_msg.edit_text(response, parse_mode='Markdown')
    except Exception as e:
        await status_msg.edit_text(f"❌ Errore di sistema: {str(e)}")

# --- SEZIONE 4: AVVIO E PROTOCOLLI DI STABILITÀ ---

def main():
    # Avvio micro-server Flask in un thread dedicato
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info("Server Flask di monitoraggio avviato.")

    # Inizializzazione Bot Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN mancante!")
        return

    application = Application.builder().token(token).build()

    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Avvio polling con gestione pulita
    logger.info("Avvio polling Telegram...")
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
