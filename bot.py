"""
╔══════════════════════════════════════════════════════════════════╗
║        ENSEMBLE MULTI-MODEL ENGINE  — Architettura V4.0         ║
║        Telegram Bot · Deploy: Render · Backend: Python          ║
╚══════════════════════════════════════════════════════════════════╝

SEZIONI:
  [1] CONFIGURAZIONE & COSTANTI
  [2] MICRO-SERVER FLASK  (Port-binding per Render)
  [3] CLIENT ASINCRONO    (Chiamate HTTP verso i provider AI)
  [4] LOGICA DI SINTESI   (Maestro + Fallback)
  [5] HANDLER TELEGRAM    (Comandi e messaggi)
  [6] AVVIO APPLICAZIONE  (Main entrypoint con clean-shutdown)
"""

# ─────────────────────────────────────────────────────────────────
# IMPORT
# ─────────────────────────────────────────────────────────────────
import os
import asyncio
import logging
import threading
from typing import Optional

import httpx
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ─────────────────────────────────────────────────────────────────
# [1] CONFIGURAZIONE & COSTANTI
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("EnsembleBot")

# — Chiavi API (lette da variabili d'ambiente; non hardcodare mai) —
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
GOOGLE_API_KEY   = os.environ["GOOGLE_API_KEY"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
MISTRAL_API_KEY  = os.environ["MISTRAL_API_KEY"]
XAI_API_KEY      = os.environ["XAI_API_KEY"]

# — Porta per Render (default 10000) —
PORT = int(os.environ.get("PORT", 10000))

# — Timeout HTTP per ogni chiamata Expert (secondi) —
EXPERT_TIMEOUT = 45
MASTER_TIMEOUT = 60


# ─────────────────────────────────────────────────────────────────
# [2] MICRO-SERVER FLASK  ·  Risolve "Port Scan Timeout" su Render
# ─────────────────────────────────────────────────────────────────
#
# Render si aspetta che un "Web Service" apra una porta HTTP.
# Il polling di Telegram non lo fa nativamente, quindi lanciamo
# Flask su un thread daemon separato: risponde 200 a ogni GET /
# e a /health, tenendo il container vivo senza interferire con
# il loop asyncio del bot.
# ─────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)


@flask_app.route("/", methods=["GET"])
@flask_app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "engine": "Ensemble V4.0"}, 200


def run_flask() -> None:
    """Avvia Flask in modalità silenziosa su $PORT."""
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    flask_app.run(host="0.0.0.0", port=PORT)


def start_flask_thread() -> None:
    thread = threading.Thread(target=run_flask, daemon=True, name="FlaskHealthServer")
    thread.start()
    logger.info(f"Flask health-server avviato sulla porta {PORT}")


# ─────────────────────────────────────────────────────────────────
# [3] CLIENT ASINCRONO  ·  Chiamate parallele ai provider AI
# ─────────────────────────────────────────────────────────────────
#
# Ogni funzione `_call_*` è una coroutine autonoma che:
#   • costruisce il payload per il proprio provider
#   • usa httpx.AsyncClient con timeout dedicato
#   • in caso di eccezione (timeout, 4xx, 5xx, errore di rete)
#     logga l'errore e restituisce None (fault tolerance silenziosa)
#
# Tutte le coroutine Expert vengono lanciate con asyncio.gather()
# → il tempo totale di attesa = tempo del modello più lento.
# ─────────────────────────────────────────────────────────────────

async def _call_groq(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    """Llama 3.3 70B via Groq — ottimizzato per velocità."""
    try:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=EXPERT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"[Expert:Groq] Fallito → {type(e).__name__}: {e}")
        return None


async def _call_deepseek(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    """DeepSeek V3 (deepseek-chat) — capacità logico-matematiche."""
    try:
        resp = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 1024,
            },
            timeout=EXPERT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"[Expert:DeepSeek] Fallito → {type(e).__name__}: {e}")
        return None


async def _call_mistral(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    """Mistral Large — ragionamento strutturato e coerenza logica."""
    try:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.65,
                "max_tokens": 1024,
            },
            timeout=EXPERT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"[Expert:Mistral] Fallito → {type(e).__name__}: {e}")
        return None


async def _call_grok(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    """Grok (xAI) — prospettive alternative e dati in tempo reale."""
    try:
        resp = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={
                "model": "grok-3-latest",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=EXPERT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"[Expert:Grok] Fallito → {type(e).__name__}: {e}")
        return None


async def gather_expert_opinions(prompt: str) -> dict[str, str]:
    """
    Interroga tutti gli Expert in parallelo con asyncio.gather.
    Restituisce un dizionario {nome_modello: risposta} filtrando i None.
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _call_groq(prompt, client),
            _call_deepseek(prompt, client),
            _call_mistral(prompt, client),
            _call_grok(prompt, client),
            return_exceptions=False,  # le eccezioni sono già gestite internamente
        )

    labels = ["Llama-3.3-70B (Groq)", "DeepSeek-V3", "Mistral-Large", "Grok (xAI)"]
    opinions = {
        label: result
        for label, result in zip(labels, results)
        if result is not None  # scarta i modelli che hanno fallito
    }

    logger.info(f"Expert rispondenti: {list(opinions.keys())}")
    return opinions


# ─────────────────────────────────────────────────────────────────
# [4] LOGICA DI SINTESI  ·  Maestro (Gemini) + Fallback (DeepSeek)
# ─────────────────────────────────────────────────────────────────
#
# Il Modello Maestro riceve:
#   • la domanda originale dell'utente
#   • i pareri grezzi degli Expert sopravvissuti
# Ha il compito di analizzarli, trovare consensus, risolvere
# contraddizioni e produrre una risposta finale autorevole.
# ─────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """Sei il Sintetizzatore di un sistema Ensemble Multi-Modello.
Hai ricevuto la domanda originale dell'utente e i pareri indipendenti di più modelli AI esperti.
Il tuo compito:
1. Analizza i pareri, individua i punti di accordo e le divergenze.
2. Costruisci una risposta definitiva, autorevole e ben strutturata.
3. Integra le prospettive complementari; ignora o segnala gli errori evidenti.
4. Non menzionare i nomi dei modelli Expert né i dettagli del processo interno.
5. Rispondi direttamente all'utente in modo chiaro e completo."""


def _build_synthesis_prompt(user_query: str, opinions: dict[str, str]) -> str:
    """Costruisce il mega-prompt da inviare al Modello Maestro."""
    opinions_block = "\n\n".join(
        f"--- Parere di {name} ---\n{text}" for name, text in opinions.items()
    )
    return (
        f"DOMANDA ORIGINALE DELL'UTENTE:\n{user_query}\n\n"
        f"PARERI DEGLI ESPERTI:\n{opinions_block}\n\n"
        "Ora produci la risposta sintetica finale."
    )


async def _synthesize_with_gemini(synthesis_prompt: str) -> Optional[str]:
    """Modello Maestro: Gemini 1.5 Flash via Google Generative Language API."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GOOGLE_API_KEY}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYNTHESIS_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": synthesis_prompt}]}],
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.5},
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=MASTER_TIMEOUT)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"[Master:Gemini] Fallito → {type(e).__name__}: {e}")
        return None


async def _synthesize_with_deepseek_fallback(synthesis_prompt: str) -> Optional[str]:
    """Sintetizzatore di Riserva: DeepSeek V3 (attivato solo se Gemini crasha)."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": synthesis_prompt},
                    ],
                    "temperature": 0.5,
                    "max_tokens": 2048,
                },
                timeout=MASTER_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"[Fallback:DeepSeek] Fallito → {type(e).__name__}: {e}")
        return None


async def run_ensemble_engine(user_query: str) -> str:
    """
    Pipeline principale del motore Ensemble:
      1. Raccoglie i pareri degli Expert in parallelo.
      2. Se nessun Expert risponde → errore informativo.
      3. Invia tutto al Modello Maestro (Gemini).
      4. Se Gemini fallisce → attiva il Fallback (DeepSeek).
      5. Se anche il Fallback fallisce → restituisce i pareri grezzi.

    NOTA: questa funzione è completamente disaccoppiata da Telegram
    e può essere esposta direttamente come endpoint REST in Fase 2.
    """
    # Step 1 — Interrogazione parallela degli Expert
    opinions = await gather_expert_opinions(user_query)

    if not opinions:
        return (
            "⚠️ Nessun modello Expert ha risposto in questo momento. "
            "Riprova tra qualche istante."
        )

    # Step 2 — Costruzione del prompt di sintesi
    synthesis_prompt = _build_synthesis_prompt(user_query, opinions)

    # Step 3 — Sintesi con Gemini (Modello Maestro)
    final_response = await _synthesize_with_gemini(synthesis_prompt)

    # Step 4 — Fallback su DeepSeek se Gemini ha fallito
    if final_response is None:
        logger.warning("Gemini non disponibile. Attivo il Fallback DeepSeek.")
        final_response = await _synthesize_with_deepseek_fallback(synthesis_prompt)

    # Step 5 — Ultimo resort: mostra i pareri grezzi concatenati
    if final_response is None:
        logger.error("Anche il Fallback ha fallito. Restituzione pareri grezzi.")
        raw = "\n\n".join(
            f"*{name}*:\n{text}" for name, text in opinions.items()
        )
        final_response = (
            "⚠️ Il sintetizzatore non è disponibile. "
            "Ecco i pareri grezzi dei modelli:\n\n" + raw
        )

    return final_response


# ─────────────────────────────────────────────────────────────────
# [5] HANDLER TELEGRAM  ·  Comandi e messaggi utente
# ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /start — benvenuto e istruzioni."""
    text = (
        "👋 *Benvenuto nel Motore Ensemble V4.0!*\n\n"
        "Sono un sistema di intelligenza collettiva che interroga "
        "simultaneamente più modelli AI e sintetizza le loro risposte "
        "in un'unica risposta autorevole.\n\n"
        "🤖 *Expert attivi:* Llama 3.3, DeepSeek V3, Mistral Large, Grok\n"
        "🎯 *Sintetizzatore:* Gemini 1.5 Flash\n\n"
        "Inviami qualsiasi domanda per iniziare!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /help — panoramica comandi."""
    text = (
        "📖 *Comandi disponibili:*\n\n"
        "/start — Messaggio di benvenuto\n"
        "/help  — Questo messaggio\n"
        "/ping  — Verifica che il bot sia attivo\n\n"
        "Per tutto il resto, scrivi liberamente e il motore Ensemble risponderà."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler /ping — health check rapido."""
    await update.message.reply_text("🟢 Pong! Il motore Ensemble è operativo.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler principale: riceve ogni messaggio di testo,
    avvia il motore Ensemble e restituisce la risposta sintetica.
    """
    user_query = update.message.text.strip()
    if not user_query:
        return

    # Feedback immediato all'utente (il processo può richiedere 10-45s)
    thinking_msg = await update.message.reply_text(
        "⚙️ *Consultando il Consiglio degli Esperti…*\n"
        "_Questo può richiedere fino a 45 secondi._",
        parse_mode="Markdown",
    )

    try:
        response = await run_ensemble_engine(user_query)
    except Exception as e:
        logger.exception(f"Errore imprevisto nel motore Ensemble: {e}")
        response = "❌ Si è verificato un errore interno. Riprova tra poco."

    # Rimuovi il messaggio "thinking" e invia la risposta finale
    await thinking_msg.delete()

    # Telegram ha un limite di 4096 caratteri per messaggio
    if len(response) <= 4096:
        await update.message.reply_text(response, parse_mode="Markdown")
    else:
        # Suddivisione sicura su boundary di paragrafo
        chunks = _split_message(response, max_len=4000)
        for chunk in chunks:
            await update.message.reply_text(chunk, parse_mode="Markdown")


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Divide un testo lungo in chunks rispettando i paragrafi."""
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_len:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para) if current else para
    if current:
        chunks.append(current.strip())
    return chunks


# ─────────────────────────────────────────────────────────────────
# [6] AVVIO APPLICAZIONE  ·  Main entrypoint con clean-shutdown
# ─────────────────────────────────────────────────────────────────
#
# PROBLEMA "Conflict: terminated by other getUpdates request":
# Si verifica quando Render riavvia il container e una vecchia
# istanza del polling è ancora attiva su Telegram. La soluzione:
#   • drop_pending_updates=True  → ignora gli update accumulati
#     durante il downtime (evita code stantie e conflitti).
#   • allowed_updates=[]         → delega a Telegram il filtro
#     sugli update rilevanti.
#   • signal handler nativo di PTB → gestisce SIGINT/SIGTERM
#     con uno shutdown pulito che chiude la sessione HTTP
#     verso i server Telegram prima di terminare.
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Avvia il micro-server Flask su thread daemon
    start_flask_thread()

    # 2. Costruisci l'applicazione Telegram
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    # 3. Registra i handler
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("ping", cmd_ping))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # 4. Avvia il polling con protezione anti-conflitto
    logger.info("Avvio del polling Telegram (Ensemble V4.0)…")
    application.run_polling(
        drop_pending_updates=True,   # ← risolve conflitti al riavvio
        allowed_updates=Update.ALL_TYPES,
        close_loop=False,            # lascia gestire il loop a PTB
    )


if __name__ == "__main__":
    main()
