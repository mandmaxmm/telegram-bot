import os, asyncio, logging, threading
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Caricamento Variabili e Configurazione Logging
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SERVER FLASK (Necessario per mantenere vivo il servizio su Render) ---
app = Flask(__name__)
@app.route('/')
def health(): return "Engine V4.6 Corazzato Online", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# --- MOTORE DI INTELLIGENZA (Ensemble Core) ---
class EnsembleEngine:
    def __init__(self):
        # Aumentato il timeout a 30s per gestire i rallentamenti di OpenRouter/Mistral
        self.timeout = httpx.Timeout(30.0, connect=10.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY")
        }

    async def call_openai_compat(self, client, url, key, model, prompt, provider):
        """Gestore universale per Groq, Mistral, OpenRouter"""
        if not key: return ""
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            if provider == "OpenRouter":
                headers.update({"HTTP-Referer": "https://render.com", "X-Title": "EnsembleBot"})
            
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            
            logger.error(f"Errore {provider} ({model}): {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ex {provider}: {e}")
            return ""

    async def call_gemini_native(self, client, prompt):
        """Chiamata nativa Google con Doppio Tentativo (v1beta e v1)"""
        if not self.keys['gemini']: return ""
        
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        # Tentativo 1: v1beta
        endpoints = [
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}",
            f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
        ]

        for url in endpoints:
            try:
                resp = await client.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()['candidates'][0]['content']['parts'][0]['text']
                logger.warning(f"Gemini endpoint {url.split('/')[3]} fallito: {resp.status_code}")
            except Exception as e:
                logger.error(f"Ex Gemini su {url.split('/')[3]}: {e}")
        
        return ""

    async def get_response(self, query):
        async with httpx.AsyncClient() as client:
            # RACCOLTA PARERI (Esperti)
            tasks = [
                self.call_openai_compat(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query, "Groq"),
                self.call_openai_compat(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query, "Mistral"),
                self.call_openai_compat(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "meta-llama/llama-3.2-3b-instruct:free", query, "OpenRouter")
            ]
            
            resps = await asyncio.gather(*tasks)
            valid = [r for r in resps if r]
            
            if not valid:
                return "📵 Tutti i servizi AI sono momentaneamente offline o sovraccarichi. Riprova tra un istante."

            # SINTESI FINALE
            context = "\n\n".join([f"Esperto {i+1}: {r}" for i, r in enumerate(valid)])
            summary_prompt = f"Sei un assistente intelligente. Crea una risposta unica e armoniosa basandoti su questi pareri:\n\n{context}"

            # Prova la sintesi con Gemini
            final = await self.call_gemini_native(client, summary_prompt)
            
            if final:
                return final
            else:
                # Se Gemini (Sintetizzatore) fallisce, restituiamo il miglior parere disponibile
                return f"{valid[0]}\n\n(Nota: Sintesi non disponibile, risposta singola Llama/Mistral)"

# --- TELEGRAM HANDLER ---
engine = EnsembleEngine()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    
    # Messaggio di stato iniziale
    status_msg = await update.message.reply_text("🤔 Consulto il pool di esperti...")
    
    try:
        response = await engine.get_response(update.message.text)
        await status_msg.edit_text(response)
    except Exception as e:
        logger.error(f"Errore Handler: {e}")
        await status_msg.edit_text("💥 Errore critico nel motore. Riprova.")

# --- AVVIO APPLICAZIONE ---
async def main():
    # Avvio Server Flask in un thread separato
    threading.Thread(target=run_flask, daemon=True).start()
    
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("ERRORE: Variabile TELEGRAM_TOKEN non trovata!")
        return

    # Configurazione Applicazione Telegram
    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot Ensemble V4.6 in fase di avvio...")
    
    async with application:
        await application.initialize()
        await application.start()
        # drop_pending_updates pulisce la coda di messaggi inviati mentre il bot era spento
        await application.updater.start_polling(drop_pending_updates=True)
        
        # Mantiene il bot in ascolto
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot spento correttamente.")
