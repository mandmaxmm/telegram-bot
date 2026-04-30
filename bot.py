import os, asyncio, logging, threading
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# --- CONFIGURAZIONE ---
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SERVER FLASK ---
app = Flask(__name__)
@app.route('/')
def health(): return "Engine V5.4 - Fix Posizionale", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# --- MOTORE MAESTRO ---
class MaestroEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(30.0, connect=10.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY")
        }

    async def call_expert(self, client, url, key, model, prompt):
        if not key: return None
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            return None
        except Exception:
            return None

    async def call_gemini(self, client, prompt):
        if not self.keys['gemini']: return None
        
        # Tentativo su v1 (stabile)
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            
            # Fallback su v1beta
            url_beta = url.replace("/v1/", "/v1beta/").replace("gemini-1.5-flash", "gemini-1.5-flash-latest")
            resp = await client.post(url_beta, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            return None
        except Exception:
            return None

    async def craft_response(self, query):
        async with httpx.AsyncClient() as client:
            # 1. Raccolta pareri (Corretta la chiamata mancante)
            tasks = [
                self.call_expert(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query),
                self.call_expert(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query)
            ]
            
            opinions = await asyncio.gather(*tasks)
            valid = [o for o in opinions if o]
            
            if not valid: return "Bottega momentaneamente chiusa. Riprova."

            # 2. Sintesi
            if len(valid) > 1:
                context = f"A: {valid[0]}\n\nB: {valid[1]}"
                synthesis_prompt = f"Sei un saggio. Crea una risposta fluida e completa basandoti su questi dati:\n\n{context}"
                final = await self.call_gemini(client, synthesis_prompt)
                if final: return final
            
            return valid[0]

# --- TELEGRAM ---
engine = MaestroEngine()

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    status = await update.message.reply_text("...")
    response = await engine.craft_response(update.message.text)
    await status.edit_text(response)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
