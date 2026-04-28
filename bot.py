import os
import logging
import asyncio
from flask import Flask
from threading import Thread
from dotenv import load_dotenv

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, 
    filters, ContextTypes, PicklePersistence
)
import google.generativeai as genai
from openai import OpenAI
from mistralai import Mistral

# --- 1. WEB SERVER ---
app_web = Flask('')
@app_web.route('/')
def home(): return "Ensemble Engine V3.1: Active & Updated"

def run():
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run); t.daemon = True; t.start()

# --- 2. SETUP ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

AUTHORIZED_USERS = [1379829807] # METTI IL TUO ID REALE QUI

# --- 3. CONFIGURAZIONE MODELLI AGGIORNATI ---
def get_ai_clients():
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    clients = {
        "gemini": genai.GenerativeModel('gemini-1.5-flash') if os.getenv("GEMINI_API_KEY") else None,
        "groq": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None,
        "mistral": Mistral(api_key=os.getenv("MISTRAL_API_KEY")) if os.getenv("MISTRAL_API_KEY") else None,
        "deepseek": OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY")) if os.getenv("DEEPSEEK_API_KEY") else None,
        "grok": OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("GROK_API_KEY")) if os.getenv("GROK_API_KEY") else None
    }
    return clients

AI = get_ai_clients()

# --- 4. LOGICA DI CHIAMATA ---
async def call_model(name, func, prompt):
    try:
        res = await func(prompt)
        return f"--- EXPERT {name.upper()} ---\n{res}"
    except Exception as e:
        logger.error(f"Errore su {name}: {e}")
        return None # Restituiamo None per filtrare i modelli offline

async def fetch_responses(prompt):
    tasks = []
    if AI["gemini"]: tasks.append(call_model("Gemini", lambda p: AI["gemini"].generate_content(p).text, prompt))
    if AI["groq"]: tasks.append(call_model("Groq", lambda p: AI["groq"].chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))
    if AI["mistral"]: tasks.append(call_model("Mistral", lambda p: AI["mistral"].chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))
    if AI["deepseek"]: tasks.append(call_model("DeepSeek", lambda p: AI["deepseek"].chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))
    if AI["grok"]: tasks.append(call_model("Grok", lambda p: AI["grok"].chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# --- 5. SINTESI ROBUSTA ---
async def synthesize(user_prompt, responses):
    if not responses: return "⚠️ Nessun modello disponibile al momento."
    
    context = "\n\n".join(responses)
    prompt_sintesi = f"Sintetizza in un'unica risposta magistrale e fluida per l'utente (domanda: {user_prompt}):\n\n{context}"
    
    # Primo tentativo: Gemini
    try:
        res = AI["gemini"].generate_content(prompt_sintesi)
        return res.text
    except:
        # Fallback: DeepSeek come sintetizzatore di riserva
        try:
            res = AI["deepseek"].chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt_sintesi}])
            return res.choices[0].message.content
        except:
            return "Sintesi fallita. Ecco i pareri grezzi:\n\n" + context

# --- 6. HANDLERS ---
async def main_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in AUTHORIZED_USERS: return
    
    user_text = update.message.text
    if user_text == '🚀 Stato Sistema':
        active = [k for k, v in AI.items() if v is not None]
        await update.message.reply_text(f"🛰️ Ensemble V3.1 Online\n🧠 Provider pronti: {', '.join(active)}")
        return

    waiting = await update.message.reply_text("🧬 Ensemble sta interrogando la squadra...")
    raw_results = await fetch_responses(user_text)
    final_answer = await synthesize(user_text, raw_results)
    
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=waiting.message_id, text=final_answer)

# --- 7. START ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if token:
        keep_alive()
        app = ApplicationBuilder().token(token).persistence(PicklePersistence(filepath="ensemble_v3.pickle")).build()
        app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Ensemble Engine Aggiornato.", reply_markup=ReplyKeyboardMarkup([['🚀 Stato Sistema']], resize_keyboard=True))))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), main_handler))
        app.run_polling()
