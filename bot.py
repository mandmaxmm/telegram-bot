import os
import asyncio
import logging
import threading
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Caricamento variabili d'ambiente
load_dotenv()

# Configurazione Logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SEZIONE 1: SERVER FLASK (Fix per Render Port Scan) ---
app = Flask(__name__)

@app.route('/')
def health(): 
    return "Ensemble Engine V4.2 (OpenRouter Enabled) Active", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

# --- SEZIONE 2: MOTORE DI INTELLIGENZA (Ensemble Core) ---
class EnsembleEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(25.0, connect=5.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "xai": os.getenv("XAI_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY")
        }

    async def call_openai_style(self, client, url, key, model, prompt, provider_name):
        """Gestore universale per API compatibili OpenAI (Groq, DeepSeek, Mistral, xAI, OpenRouter)"""
        if not key:
            return ""
        try:
            headers = {
                "Authorization": f"Bearer {key.strip()}",
                "Content-Type": "application/json"
            }
            # Header aggiuntivi richiesti specificamente da OpenRouter
            if provider_name == "OpenRouter":
                headers["HTTP-Referer"] = "https://render.com" 
                headers["X-Title"] = "Ensemble Bot V4"

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            
            logger.error(f"Errore {provider_name} ({model}): {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Eccezione {provider_name}: {e}")
            return ""

    async def call_gemini(self, client, prompt):
        """Chiamata nativa Google Gemini"""
        if not self.keys['gemini']: return ""
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Errore Gemini: {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Eccezione Gemini: {e}")
            return ""

    async def get_ensemble_response(self, query):
        async with httpx.AsyncClient() as client:
            # Pool Esperti - Esecuzione Parallela
            tasks = [
                self.call_openai_style(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query, "Groq"),
                self.call_openai_style(client, "https://api.deepseek.com/chat/completions", self.keys['deepseek'], "deepseek-chat", query, "DeepSeek"),
                self.call_openai_style(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query, "Mistral"),
                self.call_openai_style(client, "https://api.x.ai/v1/chat/completions", self.keys['xai'], "grok-2-1212", query, "xAI"),
                # OpenRouter configurato con Claude 3.5 Sonnet (uno dei migliori per logica)
                self.call_openai_style(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "anthropic/claude-3.5-sonnet", query, "OpenRouter")
            ]
            
            resps = await asyncio.gather(*tasks)
            valid_experts = [r for r in resps if r]
            
            if not valid_experts:
                return "⚠️ Nessun esperto (incluso OpenRouter) ha risposto. Verifica i crediti o le API Key."

            # Fase di Sintesi
            context = "\n\n".join([f"Esperto {i+1}: {r}" for i, r in enumerate(valid_experts)])
            synthesis_prompt = f"Sei un sintetizzatore esperto. Fondi i seguenti pareri in una risposta coerente e completa per l'utente.\nUtente: {query}\n\n{context}"

            # Maestro: Gemini -> Fallback: DeepSeek -> Ultima spiaggia: Primo esperto valido
            final = await self.call_gemini(client, synthesis_prompt)
            if not final:
                final = await self.call_openai_style(client, "https://api.deepseek.com/chat/completions", self.keys['deepseek'], "deepseek-chat", synthesis_prompt, "DeepSeek-Fallback")
            
            return final if final else f" [Sintesi Emergenza] {valid_experts[0]}"

# --- SEZIONE 3: TELEGRAM HANDLERS ---
engine = EnsembleEngine()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    
    status = await update.message.reply_text("🧠 Consultazione del pool (OpenRouter incluso)...")
    
    try:
        response = await engine.get_ensemble_response(update.message.text)
        await status.edit_text(response)
    except Exception as e:
        logger.error(f"Errore Handler: {e}")
        await status.edit_text("💥 Il motore ha avuto un sussulto. Riprova.")

# --- SEZIONE 4: BOOT ASINCRONO ---
async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN mancante!")
        return

    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        logger.info("Ensemble V4.2 Online!")
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
