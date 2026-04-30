import os, asyncio, logging, threading
from flask import Flask
import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
@app.route('/')
def health(): return "Engine V4.3 - Online", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

class EnsembleEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(20.0, connect=5.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "xai": os.getenv("XAI_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY")
        }

    async def call_api(self, client, url, key, model, prompt, provider):
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

    async def call_gemini(self, client, prompt):
        if not self.keys['gemini']: return ""
        try:
            # Proviamo v1beta con il nome modello completo
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Errore Gemini: {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ex Gemini: {e}")
            return ""

    async def get_response(self, query):
        async with httpx.AsyncClient() as client:
            tasks = [
                self.call_api(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query, "Groq"),
                self.call_api(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query, "Mistral"),
                self.call_api(client, "https://api.x.ai/v1/chat/completions", self.keys['xai'], "grok-beta", query, "xAI"),
                self.call_api(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "anthropic/claude-3", query, "OpenRouter"),
                self.call_api(client, "https://api.deepseek.com/chat/completions", self.keys['deepseek'], "deepseek-chat", query, "DeepSeek")
            ]
            resps = await asyncio.gather(*tasks)
            valid = [r for r in resps if r]
            
            if not valid: return "❌ Nessun esperto risponde (controlla crediti DeepSeek/Mistral)."

            context = "\n\n".join([f"Fonte {i+1}: {r}" for i, r in enumerate(valid)])
            summary_prompt = f"Sintetizza in modo chiaro e amichevole:\n\n{context}"

            # Tenta Sintesi con i tre "Maestri" in ordine
            for method in [self.call_gemini(client, summary_prompt), 
                           self.call_api(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "anthropic/claude-3", summary_prompt, "Sintesi-OR")]:
                final = await method
                if final: return final
            
            # Se tutti i maestri falliscono, restituisci la prima risposta valida degli esperti (Groq/Mistral)
            return f" [Sintesi Rapida] {valid[0]}"

engine = EnsembleEngine()

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.text: return
    m = await update.message.reply_text("⏳ Consulto i modelli...")
    res = await engine.get_response(update.message.text)
    await m.edit_text(res)

async def main():
    threading.Thread(target=run_flask, daemon=True).start()
    token = os.getenv("TELEGRAM_TOKEN")
    app_tg = Application.builder().token(token).build()
    app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    async with app_tg:
        await app_tg.initialize()
        await app_tg.start()
        await app_tg.updater.start_polling(drop_pending_updates=True)
        while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
