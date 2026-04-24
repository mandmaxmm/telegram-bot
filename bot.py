import os
import logging
from flask import Flask
from threading import Thread
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes, 
    PicklePersistence
)

# --- 1. CONFIGURAZIONE WEB SERVER (Per Render + UptimeRobot) ---
app_web = Flask('')

@app_web.route('/')
def home():
    return "Bot is Online and Awake!"

def run():
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- 2. LOGGING E SICUREZZA ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

# 🔴 ATTENZIONE: SOSTITUISCI 123456789 CON IL TUO ID REALE 🔴
AUTHORIZED_USERS = [1379829807] 

async def is_authorized(update: Update):
    user_id = update.effective_user.id
    if user_id not in AUTHORIZED_USERS:
        logger.warning(f"Tentativo di accesso negato per l'ID: {user_id}")
        await update.message.reply_text(f"⛔ Accesso negato. Il tuo ID ({user_id}) non è autorizzato.")
        return False
    return True

# --- 3. LOGICA DEL BOT ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    tastiera = [['🚀 Stato', '📸 Invia Foto'], ['⚙️ Impostazioni']]
    markup = ReplyKeyboardMarkup(tastiera, resize_keyboard=True)
    
    await update.message.reply_text(
        f"Ciao {update.effective_user.first_name}!\nAccesso eseguito. Il bot è ai tuoi ordini.",
        reply_markup=markup
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    await update.message.reply_text("📸 Foto ricevuta! La sto processando sul server.")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    testo = update.message.text
    if testo == '🚀 Stato':
        await update.message.reply_text("✅ Server Cloud attivo e scattante!")
    else:
        await update.message.reply_text(f"Ricevuto: {testo}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Errore: {context.error}")

# --- 4. AVVIO ---

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    
    if not token:
        logger.error("❌ TELEGRAM_TOKEN non trovato!")
    else:
        keep_alive()
        persistence = PicklePersistence(filepath="bot_data.pickle")
        app = ApplicationBuilder().token(token).persistence(persistence).build()
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
        app.add_error_handler(error_handler)
        
        print("🚀 BOT IN ASCOLTO... Verifica ID completata.")
        app.run_polling()
