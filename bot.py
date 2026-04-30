import os, asyncio, logging, threading, sys
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import httpx
from dotenv import load_dotenv

# 1. CONFIGURAZIONE LOGGING
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. SERVER FLASK PER RENDER (Health Check)
app = Flask(__name__)
@app.route('/')
def health(): return "Bot Operativo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# 3. GESTORE INTELLIGENZA ARTIFICIALE
class AIHandler:
    def __init__(self):
        load_dotenv()
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")

    async def call_gemini(self, client, prompt):
        if not self.gemini_key: return None
        # URL specifico per Google AI Studio (v1beta è il più stabile per Flash)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key.strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            r = await client.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Gemini fallito ({r.status_code}): {r.text}")
        except Exception as e:
            logger.error(f"Errore Gemini: {e}")
        return None

    async def call_openai_style(self, client, url, key, model, prompt):
        if not key: return None
        headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
        except: pass
        return None

    async def get_response(self, query):
        async with httpx.AsyncClient() as client:
            # 1. Tentativo primario: Gemini
            answer = await self.call_gemini(client, query)
            if answer: return answer
            
            # 2. Backup: Groq (Llama 3.3)
            answer = await self.call_openai_style(client, "https://api.groq.com/openai/v1/chat/completions", self.groq_key, "llama-3.3-70b-versatile", query)
            if answer: return answer
            
            # 3. Ultima spiaggia: Mistral
            answer = await self.call_openai_style(client, "https://api.mistral.ai/v1/chat/completions", self.mistral_key, "mistral-large-latest", query)
            return answer if answer else "Tutti i sistemi IA sono occupati. Riprova tra un attimo."

# 4. GESTORE MESSAGGI TELEGRAM
ai = AIHandler()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    
    # Messaggio di attesa
    status_msg = await update.message.reply_text("...")
    
    # Otteniamo la risposta dall'IA
    text_response = await ai.get_response(update.message.text)
    
    # Modifichiamo il messaggio con la risposta finale
    await status_msg.edit_text(text_response)

# 5. ESECUZIONE PRINCIPALE (Modificata per Python 3.14+)
async def start_bot():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN mancante!")
        return

    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot in fase di polling...")
    
    # Avvio pulito dell'applicazione
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        # Mantiene il bot in ascolto per sempre
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    # Avvio Flask in un thread separato per il controllo di Render
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Avvio del ciclo asincrono principale
    try:
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot arrestato.")
