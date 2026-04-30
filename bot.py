import os, asyncio, logging, threading
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import httpx
from dotenv import load_dotenv

# 1. SETUP LOGGING
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. SERVER FLASK (Per far star zitto Render)
app = Flask(__name__)
@app.route('/')
def health(): return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    # Usiamo il server di Flask in modo semplice
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# 3. GESTORE INTELLIGENZA (Gemini, Groq, Mistral)
class AIHandler:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")

    async def call_gemini(self, client, prompt):
        if not self.gemini_key: return None
        # URL e Payload specifico per Google (diverso da OpenAI)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key.strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            r = await client.post(url, json=payload, timeout=15)
            if r.status_code == 200:
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Gemini Error {r.status_code}: {r.text}")
        except Exception as e: logger.error(f"Gemini Exception: {e}")
        return None

    async def call_openai_style(self, client, url, key, model, prompt):
        if not key: return None
        headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
        except: pass
        return None

    async def get_best_answer(self, query):
        async with httpx.AsyncClient() as client:
            # Proviamo prima Gemini (ora corretto)
            ans = await self.call_gemini(client, query)
            if ans: return ans
            
            # Se Gemini fallisce, proviamo Groq o Mistral
            ans = await self.call_openai_style(client, "https://api.groq.com/openai/v1/chat/completions", self.mistral_key, "llama-3.3-70b-versatile", query)
            if ans: return ans
            
            return "Scusa, i miei cervelli sono momentaneamente offline."

# 4. LOGICA TELEGRAM
ai = AIHandler()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    # Feedback immediato all'utente
    status_msg = await update.message.reply_text("...")
    
    answer = await ai.get_best_answer(update.message.text)
    await status_msg.edit_text(answer)

# 5. AVVIO PRINCIPALE
if __name__ == "__main__":
    load_dotenv()
    
    # Avviamo Flask in un thread separato
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Avviamo il Bot
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Sistema avviato con successo.")
    application.run_polling(drop_pending_updates=True)
