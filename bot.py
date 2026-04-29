import os
import asyncio
import logging
import threading
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
@app.route('/')
def health(): return "Ensemble Engine V4.1 Active", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

class EnsembleEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(25.0, connect=5.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "xai": os.getenv("XAI_API_KEY")
        }

    async def call_openai(self, client, url, key, model, prompt):
        if not key: return ""
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            logger.error(f"Errore {model}: {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ex {model}: {e}")
            return ""

    async def call_gemini(self, client, prompt):
        if not self.keys['gemini']: return ""
        try:
            # Endpoint aggiornato alla v1 stabile
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Errore Gemini: {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ex Gemini: {e}")
            return ""

    async def get_ensemble_response(self, query):
        async with httpx.AsyncClient() as client:
            # Pool esperti
            tasks = [
                self.call_openai(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query),
                self.call_openai(client, "https://api.deepseek.com/chat/completions", self.keys['deepseek'], "deepseek-chat", query),
                self.call_openai(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query),
                self.call_openai(client, "https://api.x.ai/v1/chat/completions", self.keys['xai'], "grok-2-1212", query) # Nome modello Grok aggiornato
            ]
            resps = await asyncio.gather(*tasks)
            valid = [r for r in resps if r]
            
            if not valid: return "Nessun esperto ha risposto. Controlla i crediti delle API."

            context = "\n\n".join([f"Esperto {i+1}: {r}" for i, r in enumerate(valid)])
            prompt = f"Sintetizza in modo colloquiale:\n\n{context}"

            # Prova Maestro (Gemini)
            final = await self.call_gemini(client, prompt)
            
            # Se Gemini fallisce, prova Fallback (DeepSeek)
            if not final:
                final = await self.call_openai(client, "https://api.deepseek.com/chat/completions", self.keys['deepseek'], "deepseek-chat", prompt)
            
            # Se tutto fallisce, usa l'Expert numero 1 invece di dare errore (Sintesi d'emergenza)
            return final if final else f"[Sintesi d'emergenza]\n{valid[0]}"

engine = EnsembleEngine()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.text: return
    status = await update.message.reply_text("🤔 Rifletto...")
    res = await engine.get_ensemble_response(update.message.text)
    await status.edit_text(res)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
