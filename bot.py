import os, asyncio, logging, threading
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import httpx
from dotenv import load_dotenv

# 1. SETUP LOGGING E AMBIENTE
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. SERVER FLASK (Mantiene il servizio "vivo" per Render)
app = Flask(__name__)
@app.route('/')
def health(): return "Bot Online", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# 3. MOTORE DI INTELLIGENZA (Gestione API)
class MaestroEngine:
    def __init__(self):
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY")
        }

    async def fetch_expert(self, client, url, key, model, prompt):
        if not key: return None
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            resp = await client.post(url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
        except: pass
        return None

    async def get_response(self, query):
        async with httpx.AsyncClient() as client:
            # Chiamiamo Groq e Mistral in parallelo
            tasks = [
                self.fetch_expert(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query),
                self.fetch_expert(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query)
            ]
            results = await asyncio.gather(*tasks)
            valid = [r for r in results if r]
            
            return valid[0] if valid else "Servizio momentaneamente non disponibile."

# 4. LOGICA BOT
engine = MaestroEngine()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    msg = await update.message.reply_text("...")
    response = await engine.get_response(update.message.text)
    await msg.edit_text(response)

def main():
    # Avvia Flask prima di tutto
    threading.Thread(target=run_flask, daemon=True).start()
    
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot in esecuzione...")
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
