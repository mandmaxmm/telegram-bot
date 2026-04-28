import os
import logging
import asyncio
from flask import Flask
from threading import Thread
from dotenv import load_dotenv

# Telegram & AI Libraries
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, 
    filters, ContextTypes, PicklePersistence
)
import google.generativeai as genai
from openai import OpenAI
from mistralai import Mistral

# --- 1. CONFIGURAZIONE WEB SERVER (Keep-Alive) ---
app_web = Flask('')
@app_web.route('/')
def home(): return "Ensemble Engine: Grok Integrated"

def run():
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- 2. SETUP LOGGING & ENVIRONMENT ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Sostituisci con il tuo ID reale
AUTHORIZED_USERS = [1379829807] 

# --- 3. INIZIALIZZAZIONE CLIENT AI ---
def get_ai_clients():
    clients = {
        "gemini": genai.GenerativeModel('gemini-pro') if os.getenv("GEMINI_API_KEY") else None,
        "groq": OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None,
        "mistral": Mistral(api_key=os.getenv("MISTRAL_API_KEY")) if os.getenv("MISTRAL_API_KEY") else None,
        "deepseek": OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY")) if os.getenv("DEEPSEEK_API_KEY") else None,
        "grok": OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("GROK_API_KEY")) if os.getenv("GROK_API_KEY") else None,
        "openrouter": OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY")) if os.getenv("OPENROUTER_API_KEY") else None
    }
    if os.getenv("GEMINI_API_KEY"): genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return clients

AI = get_ai_clients()

# --- 4. CHIAMATE AI OTTIMIZZATE ---

async def call_model(name, func, prompt):
    try:
        return await func(prompt)
    except Exception as e:
        logger.error(f"Errore su {name}: {e}")
        return None

async def fetch_all_responses(prompt):
    tasks = []
    # Gemini
    if AI["gemini"]:
        tasks.append(call_model("Gemini", lambda p: AI["gemini"].generate_content(p).text, prompt))
    # Groq
    if AI["groq"]:
        tasks.append(call_model("Groq", lambda p: AI["groq"].chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))
    # Mistral
    if AI["mistral"]:
        tasks.append(call_model("Mistral", lambda p: AI["mistral"].chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))
    # DeepSeek
    if AI["deepseek"]:
        tasks.append(call_model("DeepSeek", lambda p: AI["deepseek"].chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))
    # Grok (xAI)
    if AI["grok"]:
        tasks.append(call_model("Grok", lambda p: AI["grok"].chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": p}]).choices[0].message.content, prompt))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

# --- 5. LOGICA DI SINTESI ---

async def synthesize(user_prompt, responses):
    if not responses: return "La squadra è stanca. Nessun modello ha risposto!"
    
    context = "\n\n".join([f"ANALISI MODELLO:\n{r}" for r in responses])
    prompt_sintesi = f"Sei un Supervisore AI di alto livello. Sintetizza i pareri di Gemini, Grok, DeepSeek e Mistral per: {user_prompt}\n\n{context}"
    
    try:
        res = AI["gemini"].generate_content(prompt_sintesi)
        return res.text
    except:
        return "Sintesi fallita. Ecco il primo parere disponibile:\n\n" + responses[0]

# --- 6. TELEGRAM HANDLERS ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in AUTHORIZED_USERS: return
    
    user_text = update.message.text
    if user_text == '🚀 Stato':
        active = [k for k, v in AI.items() if v is not None]
        await update.message.reply_text(f"🛰️ Sistema Ensemble Online\n🧠 Cervelli attivi: {', '.join(active)}")
        return

    waiting = await update.message.reply_text("🧬 Ensemble sta consultando Grok e la squadra...")
    raw_responses = await fetch_all_responses(user_text)
    final_answer = await synthesize(user_text, raw_responses)
    
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=waiting.message_id, text=final_answer)

# --- 7. AVVIO ---
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if token:
        keep_alive()
        persistence = PicklePersistence(filepath="ensemble_v2.pickle")
        app = ApplicationBuilder().token(token).persistence(persistence).build()
        
        app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Ensemble Engine: Grok & Co. pronti.", reply_markup=ReplyKeyboardMarkup([['🚀 Stato']], resize_keyboard=True))))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        
        app.run_polling()
