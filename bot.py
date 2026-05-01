import os, asyncio, logging, threading
from flask import Flask
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
import httpx
from dotenv import load_dotenv

# 1. CONFIGURAZIONE LOGGING
# Impostiamo il log per vedere chiaramente cosa succede nei server di Render
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. SERVER FLASK (Necessario per Render)
# Questo pezzo di codice dice a Render: "Ehi, sono vivo e sto ascoltando sulla porta 10000"
app = Flask(__name__)

@app.route('/')
def health_check():
    return "Bot is running!", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    # Usiamo il server integrato di Flask in modalità thread-safe
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# 3. MOTORE DI INTELLIGENZA ARTIFICIALE (Multi-Modello)
class AIHandler:
    def __init__(self):
        load_dotenv()
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.mistral_key = os.getenv("MISTRAL_API_KEY")

    async def call_gemini(self, client, prompt):
        """Tenta di chiamare Gemini usando la versione stabile o la beta come backup."""
        if not self.gemini_key: return None
        
        # Primo tentativo: Versione v1 (Stabile)
        url_v1 = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.gemini_key.strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            r = await client.post(url_v1, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json()['candidates'][0]['content']['parts'][0]['text']
            
            # Secondo tentativo se il primo dà 404: v1beta con suffisso -latest
            if r.status_code == 404:
                url_beta = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.gemini_key.strip()}"
                r = await client.post(url_beta, json=payload, timeout=20)
                if r.status_code == 200:
                    return r.json()['candidates'][0]['content']['parts'][0]['text']
            
            logger.error(f"Gemini Error {r.status_code}: {r.text}")
        except Exception as e:
            logger.error(f"Eccezione Gemini: {e}")
        return None

    async def call_openai_style(self, client, url, key, model, prompt):
        """Gestisce le API di Groq e Mistral (formato standard OpenAI)."""
        if not key: return None
        headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        try:
            r = await client.post(url, json=payload, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Errore su {model}: {e}")
        return None

    async def get_response(self, query):
        """Prova i modelli in ordine di priorità: Gemini -> Groq -> Mistral."""
        async with httpx.AsyncClient() as client:
            # 1. Priorità: Gemini
            answer = await self.call_gemini(client, query)
            if answer: return answer
            
            # 2. Backup 1: Groq (Llama 3.3 è velocissimo)
            answer = await self.call_openai_style(client, "https://api.groq.com/openai/v1/chat/completions", self.groq_key, "llama-3.3-70b-versatile", query)
            if answer: return answer
            
            # 3. Backup 2: Mistral
            answer = await self.call_openai_style(client, "https://api.mistral.ai/v1/chat/completions", self.mistral_key, "mistral-large-latest", query)
            return answer if answer else "⚠️ Tutti i servizi IA sono momentaneamente offline."

# 4. LOGICA DEL BOT TELEGRAM
ai_engine = AIHandler()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Ignora messaggi vuoti o non testuali
    if not update.message or not update.message.text:
        return

    # Invia un messaggio di caricamento (molto utile per l'utente)
    status_msg = await update.message.reply_text("...")
    
    # Ottieni la risposta dall'intelligenza artificiale
    user_query = update.message.text
    response = await ai_engine.get_response(user_query)
    
    # Modifica il messaggio precedente con la risposta finale
    try:
        await status_msg.edit_text(response)
    except Exception as e:
        logger.error(f"Errore modifica messaggio: {e}")
        await update.message.reply_text(response)

# 5. AVVIO DEL SISTEMA (Compatibile con Python 3.14+)
async def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("ERRORE: TELEGRAM_TOKEN non trovato nelle variabili d'ambiente!")
        return

    # Configurazione dell'applicazione Telegram
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot in ascolto...")

    # Gestione corretta del ciclo di vita per evitare il RuntimeError su Render
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        # Questo mantiene il bot attivo all'infinito
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    # Avviamo il server Flask in un thread secondario per non bloccare Telegram
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Avviamo il bot Telegram usando il nuovo standard asincrono
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot spento correttamente.")
