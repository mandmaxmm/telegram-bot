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

# Librerie AI
import google.generativeai as genai
from openai import OpenAI

# --- 1. CONFIGURAZIONE WEB SERVER (Keep-Alive) ---
app_web = Flask('')
@app_web.route('/')
def home(): return "Ensemble Multimodale Online!"

def run():
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- 2. SETUP LOGGING E SICUREZZA ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

AUTHORIZED_USERS = [1379829807] # Sostituisci col tuo ID

# --- 3. CONFIGURAZIONE PROVIDER AI ---

# Gemini (Google)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

# Groq (Llama/Mistral alta velocità)
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
) if os.getenv("GROQ_API_KEY") else None

# OpenRouter (Per DeepSeek, Mistral, Claude)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
) if os.getenv("OPENROUTER_API_KEY") else None

# --- 4. LOGICA ENSEMBLE (MULTI-THREADING) ---

async def call_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return f"--- EXPERT GEMINI ---\n{response.text}"
    except Exception as e: return f"Gemini Error: {e}"

async def call_groq(prompt):
    if not groq_client: return "Groq non configurato."
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return f"--- EXPERT GROQ (LLAMA3) ---\n{response.choices[0].message.content}"
    except Exception as e: return f"Groq Error: {e}"

async def call_openrouter(prompt):
    if not openrouter_client: return "OpenRouter non configurato."
    try:
        # Qui usiamo DeepSeek tramite OpenRouter
        response = openrouter_client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return f"--- EXPERT DEEPSEEK ---\n{response.choices[0].message.content}"
    except Exception as e: return f"OpenRouter Error: {e}"

async def synthesize_responses(user_prompt, results):
    """Prende tutti i pareri e crea la sintesi finale usando Gemini."""
    context = "\n\n".join(results)
    prompt_sintesi = f"""
    Sei il 'Sintetizzatore Ensemble'. Hai ricevuto diverse analisi per la domanda: "{user_prompt}"
    
    Analisi ricevute:
    {context}
    
    Compito: Crea una risposta magistrale che integri i punti chiave di tutti gli esperti. 
    Sii tecnico ma chiaro, elimina le ripetizioni e correggi eventuali incongruenze tra i modelli.
    """
    try:
        sintesi = gemini_model.generate_content(prompt_sintesi)
        return sintesi.text
    except:
        return "Errore nella fase di sintesi. Ecco i pareri grezzi:\n\n" + context

# --- 5. HANDLERS TELEGRAM ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in AUTHORIZED_USERS: return
    tastiera = [['🚀 Stato Sistema', '🧠 Reset Memoria']]
    markup = ReplyKeyboardMarkup(tastiera, resize_keyboard=True)
    await update.message.reply_text(
        "🛠️ Ensemble Engine Attivo.\nGemini, Groq e DeepSeek sono pronti.\nCosa analizziamo?",
        reply_markup=markup
    )

async def main_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in AUTHORIZED_USERS: return
    
    user_prompt = update.message.text
    if user_prompt == '🚀 Stato Sistema':
        msg = "📡 Connessioni:\n"
        msg += "✅ Gemini: OK\n"
        msg += "✅ Groq: " + ("OK" if groq_client else "OFF") + "\n"
        msg += "✅ OpenRouter: " + ("OK" if openrouter_client else "OFF")
        await update.message.reply_text(msg)
        return

    waiting_msg = await update.message.reply_text("🧬 Ensemble sta interrogando gli esperti...")

    # Lancio parallelo di tutti i modelli per risparmiare tempo
    tasks = [
        call_gemini(user_prompt),
        call_groq(user_prompt),
        call_openrouter(user_prompt)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Generazione sintesi
    final_response = await synthesize_responses(user_prompt, results)

    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=waiting_msg.message_id,
        text=final_response
    )

# --- 6. AVVIO ---

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("Manca il token Telegram!")
    else:
        keep_alive()
        persistence = PicklePersistence(filepath="bot_data.pickle")
        app = ApplicationBuilder().token(token).persistence(persistence).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), main_handler))
        print("🚀 ENSEMBLE MULTI-MODEL CLOUD ONLINE...")
        app.run_polling()
