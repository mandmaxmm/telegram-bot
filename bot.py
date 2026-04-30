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
def health(): return "Engine V5.1 Maestro - Solid", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

class MaestroEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(20.0, connect=5.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY")
        }

    async def call_expert(self, client, url, key, model, prompt, name):
        if not key: return None
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            return None
        except Exception:
            return None

    async def call_gemini(self, client, prompt):
        """Metodo blindato: prova diversi nomi modello per evitare il 404"""
        if not self.keys['gemini']: return None
        
        # Lista prioritaria di modelli validi per la regione Oregon/Global
        models = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash-exp"]
        
        for model_name in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.keys['gemini'].strip()}"
            try:
                resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()['candidates'][0]['content']['parts'][0]['text']
            except Exception:
                continue
        return None

    async def craft_response(self, query):
        async with httpx.AsyncClient() as client:
            # Chiamate simultanee ai due saggi
            tasks = [
                self.call_expert(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query, "Groq"),
                self.call_expert(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query, "Mistral")
            ]
            
            results = await asyncio.gather(*tasks)
            valid = [r for r in results if r]
            
            if not valid:
                return "Bottega chiusa: i servizi AI non rispondono. Riprova tra poco."

            # Se abbiamo pareri multipli, Gemini sintetizza. Altrimenti usiamo il migliore.
            if len(valid) > 1:
                context = f"Parere A: {valid[0]}\n\nParere B: {valid[1]}"
                synthesis_prompt = f"Unisci questi pareri in una risposta coerente, elegante e senza ripetizioni:\n\n{context}"
                final = await self.call_gemini(client, synthesis_prompt)
                if final: return final
            
            return valid[0]

engine = MaestroEngine()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    
    # Risposta immediata per feedback utente
    status_msg = await update.message.reply_text("...")
    
    response = await engine.craft_response(update.message.text)
    await status_msg.edit_text(response)

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
