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

# --- 1. CONFIGURAZIONE WEB SERVER (Per l'indipendenza su Render) ---
app_web = Flask('')

@app_web.route('/')
def home():
    return "Bot Telegram Online!"

def run():
    # Render assegna una porta automaticamente, di default usiamo la 8080
    port = int(os.environ.get("PORT", 8080))
    app_web.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- 2. CONFIGURAZIONE LOGGING E SICUREZZA ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv(".env")

# SOSTITUISCI CON IL TUO ID TELEGRAM REALE
AUTHORIZED_USERS = [1379829807] 

async def is_authorized(update: Update):
    if update.effective_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("⛔ Accesso negato. ID non autorizzato.")
        return False
    return True

# --- 3. HANDLERS DEL BOT ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    # Menu a pulsanti per facilità d'uso da mobile/browser
    tastiera = [['🚀 Stato', '📸 Invia Foto'], ['⚙️ Impostazioni']]
    markup = ReplyKeyboardMarkup(tastiera, resize_keyboard=True)
    
    await update.message.reply_text(
        f"Ciao {update.effective_user.first_name}!\nIl bot è ora pronto per il Cloud.",
        reply_markup=markup
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    await update.message.reply_text("Ho ricevuto la tua foto! È pronta per essere elaborata.")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    
    testo = update.message.text
    if testo == '🚀 Stato':
        await update.message.reply_text("✅ Sistema operativo. Pronto per il deployment!")
    elif testo == '📸 Invia Foto':
        await update.message.reply_text("Ricevuto! Mandami pure un'immagine.")
    else:
        await update.message.reply_text(f"Ricevuto: {testo}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Errore rilevato: {context.error}")

# --- 4. AVVIO APPLICAZIONE ---

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    
    if not token:
        print("❌ ERRORE: TELEGRAM_TOKEN non trovato nel file .env")
    else:
        # Avviamo il server web per il "Keep Alive" (necessario per Render)
        keep_alive()
        
        # Persistenza dati locale (salva le impostazioni anche se il bot si riavvia)
        persistence = PicklePersistence(filepath="bot_data.pickle")
        
        # Costruzione dell'App
        app = ApplicationBuilder().token(token).persistence(persistence).build()
        
        # Aggiunta Handler
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))
        app.add_error_handler(error_handler)
        
        print("🚀 BOT IN ASCOLTO... Il server web è attivo su porta 8080")
        app.run_polling()