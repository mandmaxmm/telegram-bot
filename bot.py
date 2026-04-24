import os
import logging
import asyncio
import google.generativeai as genai
from openai import OpenAI
from flask import Flask
from threading import Thread
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, 
    filters, ContextTypes, PicklePersistence
)

# --- 1. CONFIGURAZIONE WEB SERVER (Keep-Alive per Render) ---
app_web = Flask('')

@app_web.route('/')
def home():
    return "Ensemble Bot is Online and Thinking!"

def run():
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- 2. LOGGING E SICUREZZA ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# INSERISCI IL TUO ID (visto nello screenshot)
AUTHORIZED_USERS = [1379829807] 

# Configurazione AI Modelli
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-pro')

# Client OpenAI (opzionale, si attiva se aggiungi la chiave su Render)
client_openai = None
if os.getenv("OPENAI_API_KEY"):
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def is_authorized(update: Update):
    if update.effective_user.id not in AUTHORIZED_USERS:
        return False
    return True

# --- 3. LOGICA MULTI-MODELLO (ENSEMBLE & SINTESI) ---

async def get_gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Errore Gemini: {e}"

async def get_openai_response(prompt):
    if not client_openai: return None
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore OpenAI: {e}"

async def ensemble_synthesis(user_prompt):
    """Interroga i modelli e sintetizza una risposta unica."""
    # Fase 1: Raccolta pareri dagli Esperti
    risposta_gemini = await get_gemini_response(user_prompt)
    risposta_openai = await get_openai_response(user_prompt)

    if not risposta_openai:
        return risposta_gemini # Se c'è solo Gemini, restituiamo quello

    # Fase 2: Sintesi finale (Chiediamo a Gemini di unire le informazioni)
    prompt_sintesi = f"""
    Agisci come un sintetizzatore esperto. Hai ricevuto due risposte a questa domanda: "{user_prompt}"
    Risposta 1: {risposta_gemini}
    Risposta 2: {risposta_openai}
    Crea una risposta finale completa, eliminando le ripetizioni e mantenendo i punti di forza di entrambe.
    """
    sintesi_finale = await get_gemini_response(prompt_sintesi)
    return sintesi_finale

# --- 4. HANDLERS ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    tastiera = [['🚀 Stato Sistema', '🧠 Modalità Ensemble'], ['📸 Invia Foto']]
    markup = ReplyKeyboardMarkup(tastiera, resize_keyboard=True)
    await update.message.reply_text(
        "Sincronizzazione completata. Sistema Ensemble Multi-Modello attivo.\nCosa vuoi analizzare oggi?",
        reply_markup=markup
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    user_text = update.message.text
    
    if user_text == '🚀 Stato Sistema':
        status = "✅ Cloud: Online\n✅ Gemini: Attivo"
        status += "\n✅ OpenAI: Collegato" if client_openai else "\n⚠️ OpenAI: Non configurato"
        await update.message.reply_text(status)
        return

    msg_attesa = await update.message.reply_text("💎 Interrogo gli esperti e sintetizzo la risposta...")

    # Esecuzione Sintesi Ensemble
    risposta_finale = await ensemble_synthesis(user_text)

    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id,
        message_id=msg_attesa.message_id,
        text=risposta_finale
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Errore: {context.error}")

# --- 5. ESECUZIONE ---

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    
    if not token:
        logger.error("Token mancante!")
    else:
        keep_alive()
        persistence = PicklePersistence(filepath="bot_data.pickle")
        app = ApplicationBuilder().token(token).persistence(persistence).build()
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.add_error_handler(error_handler)
        
        print("🚀 ENSEMBLE CLOUD ONLINE...")
        app.run_polling()
