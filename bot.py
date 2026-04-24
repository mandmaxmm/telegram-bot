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
    # Render assegna la porta automaticamente
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- 2. LOGGING E SICUREZZA ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(".env")

# IMPORTANTE: Sostituisci con il tuo ID numerico (chiedilo a @userinfobot)
AUTHORIZED_USERS = [123456789] 

async def is_authorized(update: Update):
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("⛔ Non sei autorizzato a usare questo bot.")
        return False
    return True

# --- 3. LOGICA DEL BOT ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    tastiera = [['🚀 Stato', '📸 Invia Foto'], ['⚙️ Impostazioni']]
    markup = ReplyKeyboardMarkup(tastiera, resize_keyboard=True)
    
    await update.message.reply_text(
        f"Ciao {update.effective_user.first_name}!\nBot attivo h24 su Render.",
        reply_markup=markup
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    await update.message.reply_text("📸 Immagine ricevuta e analizzata!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    testo = update.message.text
    if testo == '🚀 Stato':
        await update.message.reply_text("✅ Tutto regolare. Il server è scattante!")
    else:
        await update.message.reply_text(f"Hai scritto: {testo}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Errore: {context.error}")

# --- 4. AVVIO ---

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    
    if not token:
        print("❌ Token mancante!")
    else:
        keep_alive() # Avvia il "battito cardiaco" web
        
        persistence = PicklePersistence(filepath="bot_data.pickle")
        app = ApplicationBuilder().token(token).persistence(persistence).build()
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
        app.add_error_handler(error_handler)
        
        print("🚀 BOT IN ASCOLTO CLOUD...")
        app.run_polling()
