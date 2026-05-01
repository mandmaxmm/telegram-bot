import os, asyncio, logging, threading
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import httpx
from dotenv import load_dotenv

# 1. LOGGING
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. FLASK (Per Render)
app = Flask(__name__)
@app.route('/')
def health(): return "Vivo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# 3. AI ENGINE
class AIHandler:
    def __init__(self):
        load_dotenv()
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")

    async def call_gemini(self, client, prompt):
        if not self.gemini_key: return None
        # Cambiato URL: proviamo v1beta con il nome modello base
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key.strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            r = await client.post(url, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Gemini fallito ({r.status_code}): {r.text}")
        except Exception as e:
            logger.error(f"Eccezione Gemini: {e}")
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
            # 1. Gemini
            ans = await self.call_gemini(client, query)
            if ans: return ans
            # 2. Groq
            ans = await self.call_openai_style(client, "https://api.groq.com/openai/v1/chat/completions", self.groq_key, "llama-3.3-70b-versatile", query)
            if ans: return ans
            # 3. Mistral
            ans = await self.call_openai_style(client, "https://api.mistral.ai/v1/chat/completions", self.mistral_key, "mistral-large-latest", query)
            return ans if ans else "Errore di connessione ai server IA."

# 4. TELEGRAM HANDLER
ai = AIHandler()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    status_msg = await update.message.reply_text("...")
    response = await ai.get_response(update.message.text)
    try:
        await status_msg.edit_text(response)
    except:
        await update.message.reply_text(response)

# 5. MAIN (Risolve il conflitto all'avvio)
async def main():
    token = os.getenv("TELEGRAM_TOKEN")
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Inizializzazione pulita
    async with application:
        await application.initialize()
        await application.start()
        # drop_pending_updates=True aiuta a pulire i messaggi accumulati durante i crash
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("Bot Online!")
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
