import os, asyncio, logging, threading
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Caricamento e Logging
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SERVER FLASK PER RENDER ---
app = Flask(__name__)
@app.route('/')
def health(): return "Engine V4.5 Active", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# --- MOTORE INTELLIGENZA ---
class EnsembleEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(20.0, connect=5.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY")
        }

    async def call_openai_compat(self, client, url, key, model, prompt, provider):
        if not key: return ""
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            if provider == "OpenRouter":
                headers.update({"HTTP-Referer": "https://render.com", "X-Title": "EnsembleBot"})
            
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            logger.error(f"Errore {provider}: {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ex {provider}: {e}")
            return ""

    async def call_gemini_native(self, client, prompt):
        if not self.keys['gemini']: return ""
        try:
            # URL Standard V1 per stabilità
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            return ""
        except Exception as e:
            logger.error(f"Ex Gemini: {e}")
            return ""

    async def get_response(self, query):
        async with httpx.AsyncClient() as client:
            # Fase 1: Raccolta pareri dagli Esperti
            tasks = [
                self.call_openai_compat(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query, "Groq"),
                self.call_openai_compat(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query, "Mistral"),
                self.call_openai_compat(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "google/gemini-2.0-flash-001", query, "OpenRouter")
            ]
            resps = await asyncio.gather(*tasks)
            valid = [r for r in resps if r]
            
            if not valid: return "⚠️ Nessun esperto è disponibile al momento."

            # Fase 2: Sintesi
            context = "\n\n".join([f"Opinione {i+1}: {r}" for i, r in enumerate(valid)])
            summary_prompt = f"Unisci queste risposte in modo colloquiale:\n\n{context}"

            # Prova Gemini, se fallisce restituisci il primo parere valido
            final = await self.call_gemini_native(client, summary_prompt)
            return final if final else valid[0]

# --- TELEGRAM ---
engine = EnsembleEngine()

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    status = await update.message.reply_text("🤔 Elaborazione...")
    response = await engine.get_response(update.message.text)
    await status.edit_text(response)

async def main():
    # Avvio Flask in thread separato
    threading.Thread(target=run_flask, daemon=True).start()
    
    token = os.getenv("TELEGRAM_TOKEN")
    if not token: 
        logger.error("Token mancante!")
        return

    # Inizializzazione Bot
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    
    logger.info("Bot in avvio...")
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        # Loop infinito per mantenere il bot attivo
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
