"""
╔══════════════════════════════════════════════════════════════════╗
║        ENSEMBLE MULTI-MODEL ENGINE  — Architettura V4.1         ║
║        Telegram Bot · Deploy: Render · Backend: Python          ║
╠══════════════════════════════════════════════════════════════════╣
║  Novità V4.1:                                                    ║
║  • Master Chain a cascata (4 candidati, nessun dump grezzo)      ║
║  • Gemini 2.0 Flash su endpoint v1 stabile                       ║
║  • OpenRouter come hub Expert opzionale (modelli free)           ║
║  • Classificazione silenziosa errori 401/402/403/429             ║
║  • Comando /status — live health check di tutti i modelli        ║
║  • Fallback finale: miglior risposta singola (no doppi grezzi)   ║
╚══════════════════════════════════════════════════════════════════╝

SEZIONI:
  [1] CONFIGURAZIONE & COSTANTI
  [2] MICRO-SERVER FLASK      (Port-binding per Render)
  [3] HELPER ERRORI HTTP      (Classificazione silenziosa)
  [4] EXPERT POOL             (Chiamate parallele ai provider)
  [5] MASTER CHAIN            (Sintetizzatori a cascata)
  [6] PIPELINE ENSEMBLE       (Orchestrazione completa)
  [7] HANDLER TELEGRAM        (Comandi e messaggi)
  [8] AVVIO APPLICAZIONE      (Main con clean-shutdown)
"""

import os
import asyncio
import logging
import threading
from typing import Optional, Callable, Awaitable

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

# Chiavi obbligatorie — il bot non parte senza queste
_REQUIRED_ENV_VARS = [
    "TELEGRAM_TOKEN",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
]

# Chiavi opzionali — se mancanti, i modelli correlati vengono saltati in silenzio
_OPTIONAL_ENV_VARS = [
    "GOOGLE_API_KEY",      # Gemini
    "XAI_API_KEY",         # Grok
    "OPENROUTER_API_KEY",  # Hub multi-modello free (fortemente consigliato)
]

_missing = [k for k in _REQUIRED_ENV_VARS if not os.environ.get(k)]
if _missing:
    print(
        "\n❌ ERRORE — Variabili d'ambiente obbligatorie mancanti:\n"
        + "\n".join(f"   • {k}" for k in _missing)
        + "\n\nVai su Render → Environment e aggiungile.\n"
    )
    raise SystemExit(1)

for _k in _OPTIONAL_ENV_VARS:
    if not os.environ.get(_k):
        logger.warning(f"Chiave opzionale '{_k}' non configurata — modelli correlati disabilitati.")

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY     = os.environ["GROQ_API_KEY"]
MISTRAL_API_KEY  = os.environ["MISTRAL_API_KEY"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

GOOGLE_API_KEY     = os.environ.get("GOOGLE_API_KEY", "")
XAI_API_KEY        = os.environ.get("XAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

PORT           = int(os.environ.get("PORT") or 10000)
EXPERT_TIMEOUT = 40
MASTER_TIMEOUT = 55


# ─────────────────────────────────────────────────────────────────
# [2] MICRO-SERVER FLASK  ·  Port-binding per Render
# ─────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
@flask_app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "engine": "Ensemble V4.1"}, 200

def run_flask() -> None:
    import logging as _log
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    flask_app.run(host="0.0.0.0", port=PORT)

def start_flask_thread() -> None:
    t = threading.Thread(target=run_flask, daemon=True, name="FlaskHealth")
    t.start()
    logger.info(f"Flask health-server avviato sulla porta {PORT}")


# ─────────────────────────────────────────────────────────────────
# [3] HELPER ERRORI HTTP  ·  Classificazione silenziosa
# ─────────────────────────────────────────────────────────────────
#
# Un modello con credito esaurito o chiave scaduta NON è un errore
# del sistema. Tutti i codici 4xx vengono loggati a WARNING e il
# modello viene accantonato silenziosamente.
# ─────────────────────────────────────────────────────────────────

_SILENT_HTTP_CODES = {401, 402, 403, 404, 429}
_HTTP_REASONS = {
    401: "chiave non valida",
    402: "credito esaurito",
    403: "accesso negato",
    404: "modello non trovato",
    429: "rate limit",
}

def _log_model_failure(label: str, exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code in _SILENT_HTTP_CODES:
            reason = _HTTP_REASONS.get(code, f"HTTP {code}")
            logger.warning(f"[{label}] Non disponibile ({reason}) — saltato.")
            return
    logger.warning(f"[{label}] Fallito — {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────
# [4] EXPERT POOL  ·  Chiamate parallele ai provider
# ─────────────────────────────────────────────────────────────────

async def _call_groq(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Expert:Groq/Llama3.3", e)
        return None

async def _call_deepseek(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        r = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.6, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Expert:DeepSeek", e)
        return None

async def _call_mistral(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    try:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": "mistral-large-latest",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.65, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Expert:Mistral", e)
        return None

async def _call_grok(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not XAI_API_KEY:
        return None
    try:
        r = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={"model": "grok-3-latest",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Expert:Grok", e)
        return None

# ── OpenRouter: hub multi-modello con tier free ───────────────────

async def _call_openrouter(model_id: str, label: str, prompt: str,
                            client: httpx.AsyncClient) -> Optional[str]:
    if not OPENROUTER_API_KEY:
        return None
    try:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://ensemble-bot.onrender.com",
                "X-Title": "Ensemble Multi-Model Bot",
            },
            json={"model": model_id,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure(f"Expert:OpenRouter/{label}", e)
        return None

async def _call_or_gemini_free(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter("google/gemini-2.0-flash-exp:free", "Gemini2.0", p, c)

async def _call_or_llama_free(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter("meta-llama/llama-3.3-70b-instruct:free", "Llama3.3", p, c)

async def _call_or_deepseek_free(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter("deepseek/deepseek-chat-v3-0324:free", "DeepSeek", p, c)

async def _call_or_qwen_free(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter("qwen/qwen-2.5-72b-instruct:free", "Qwen2.5", p, c)


async def gather_expert_opinions(prompt: str) -> dict[str, str]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _call_groq(prompt, client),
            _call_deepseek(prompt, client),
            _call_mistral(prompt, client),
            _call_grok(prompt, client),
            _call_or_gemini_free(prompt, client),
            _call_or_llama_free(prompt, client),
            _call_or_deepseek_free(prompt, client),
            _call_or_qwen_free(prompt, client),
        )
    labels = [
        "Llama 3.3 70B (Groq)",
        "DeepSeek V3",
        "Mistral Large",
        "Grok (xAI)",
        "Gemini 2.0 Flash (OpenRouter free)",
        "Llama 3.3 70B (OpenRouter free)",
        "DeepSeek V3 (OpenRouter free)",
        "Qwen 2.5 72B (OpenRouter free)",
    ]
    opinions = {l: r for l, r in zip(labels, results) if r is not None}
    logger.info(f"Expert rispondenti ({len(opinions)}): {list(opinions.keys())}")
    return opinions


# ─────────────────────────────────────────────────────────────────
# [5] MASTER CHAIN  ·  Sintetizzatori a cascata
# ─────────────────────────────────────────────────────────────────
#
# Ordine di priorità:
#   1. Gemini 2.0 Flash (Google v1 — non più v1beta)
#   2. Mistral Large
#   3. DeepSeek V3
#   4. Groq Llama 3.3  ← anchor, sempre disponibile
#
# Se TUTTI falliscono → miglior risposta singola Expert (no dump grezzi).
# ─────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = (
    "Sei il Sintetizzatore di un sistema Ensemble Multi-Modello. "
    "Hai ricevuto la domanda originale dell'utente e i pareri indipendenti "
    "di piu modelli AI esperti. Il tuo compito:\n"
    "1. Analizza i pareri, identifica accordi e divergenze.\n"
    "2. Costruisci una risposta definitiva, autorevole e ben strutturata.\n"
    "3. Integra le prospettive complementari; ignora gli errori evidenti.\n"
    "4. NON menzionare i nomi dei modelli Expert ne i dettagli interni.\n"
    "5. Rispondi direttamente all'utente in italiano, in modo chiaro e completo."
)

def _build_synthesis_prompt(user_query: str, opinions: dict[str, str]) -> str:
    block = "\n\n".join(
        f"--- Parere {i+1} ---\n{text}"
        for i, text in enumerate(opinions.values())
    )
    return (
        f"DOMANDA ORIGINALE:\n{user_query}\n\n"
        f"PARERI DEGLI ESPERTI:\n{block}\n\n"
        "Produci ora la risposta sintetica finale."
    )

async def _master_gemini(sp: str) -> Optional[str]:
    if not GOOGLE_API_KEY:
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
    )
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(url, json={
                "system_instruction": {"parts": [{"text": SYNTHESIS_SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": sp}]}],
                "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.4},
            }, timeout=MASTER_TIMEOUT)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        _log_model_failure("Master:Gemini", e)
        return None

async def _master_mistral(sp: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                json={"model": "mistral-large-latest",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user",   "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Master:Mistral", e)
        return None

async def _master_deepseek(sp: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={"model": "deepseek-chat",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user",   "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Master:DeepSeek", e)
        return None

async def _master_groq(sp: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user",   "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_model_failure("Master:Groq", e)
        return None

MASTER_CHAIN: list[tuple[str, Callable[[str], Awaitable[Optional[str]]]]] = [
    ("Gemini 2.0 Flash", _master_gemini),
    ("Mistral Large",    _master_mistral),
    ("DeepSeek V3",      _master_deepseek),
    ("Groq Llama 3.3",   _master_groq),
]

async def run_master_chain(sp: str) -> tuple[Optional[str], Optional[str]]:
    for name, fn in MASTER_CHAIN:
        result = await fn(sp)
        if result:
            logger.info(f"Sintesi completata da: {name}")
            return result, name
    return None, None


# ─────────────────────────────────────────────────────────────────
# [6] PIPELINE ENSEMBLE
# ─────────────────────────────────────────────────────────────────

def _best_single_expert(opinions: dict[str, str]) -> str:
    """Seleziona la risposta Expert più completa (proxy: n° parole)."""
    return max(opinions.values(), key=lambda t: len(t.split()))

async def run_ensemble_engine(user_query: str) -> str:
    """
    Pipeline principale — disaccoppiata da Telegram, pronta per REST in Fase 2.
    """
    opinions = await gather_expert_opinions(user_query)

    if not opinions:
        return (
            "⚠️ Nessun modello Expert ha risposto in questo momento.\n"
            "I servizi AI potrebbero essere temporaneamente irraggiungibili. "
            "Riprova tra qualche istante."
        )

    sp = _build_synthesis_prompt(user_query, opinions)
    final_response, _ = await run_master_chain(sp)

    if final_response is None:
        logger.error("Tutta la Master Chain fallita — uso miglior risposta Expert singola.")
        final_response = _best_single_expert(opinions)

    return final_response


# ─────────────────────────────────────────────────────────────────
# [7] HANDLER TELEGRAM
# ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    experts = "Llama 3.3, DeepSeek V3, Mistral Large"
    if XAI_API_KEY:
        experts += ", Grok"
    if OPENROUTER_API_KEY:
        experts += ", Gemini/Llama/DeepSeek/Qwen (OpenRouter free)"
    text = (
        f"👋 *Benvenuto nel Motore Ensemble V4.1!*\n\n"
        f"Sono un sistema di intelligenza collettiva che interroga "
        f"simultaneamente più modelli AI e fonde le loro risposte "
        f"in un'unica risposta autorevole.\n\n"
        f"🤖 *Expert attivi:* {experts}\n"
        f"🎯 *Sintetizzatori (catena):* Gemini → Mistral → DeepSeek → Groq\n\n"
        f"Inviami qualsiasi domanda, oppure usa /status per vedere lo stato dei modelli."
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "📖 *Comandi disponibili:*\n\n"
        "/start  — Benvenuto e panoramica\n"
        "/help   — Questo messaggio\n"
        "/ping   — Verifica che il bot sia attivo\n"
        "/status — Stato live di tutti i modelli AI\n\n"
        "Per tutto il resto, scrivi liberamente."
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🟢 Pong! Il motore Ensemble V4.1 è operativo.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ping leggero su ogni modello — riporta chi risponde in tempo reale."""
    probe = "Rispondi solo con OK."
    msg = await update.message.reply_text("🔍 Controllo stato modelli in corso...")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _call_groq(probe, client),
            _call_deepseek(probe, client),
            _call_mistral(probe, client),
            _call_grok(probe, client),
            _call_or_gemini_free(probe, client),
            _call_or_llama_free(probe, client),
            _call_or_deepseek_free(probe, client),
            _call_or_qwen_free(probe, client),
        )

    rows = [
        ("Llama 3.3 70B",          "Groq",       bool(GROQ_API_KEY)),
        ("DeepSeek V3",             "DeepSeek",   bool(DEEPSEEK_API_KEY)),
        ("Mistral Large",           "Mistral",    bool(MISTRAL_API_KEY)),
        ("Grok",                    "xAI",        bool(XAI_API_KEY)),
        ("Gemini 2.0 Flash (free)", "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Llama 3.3 (free)",        "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("DeepSeek (free)",         "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Qwen 2.5 72B (free)",     "OpenRouter", bool(OPENROUTER_API_KEY)),
    ]

    lines = ["🤖 *Stato modelli Expert:*\n"]
    for (name, provider, configured), result in zip(rows, results):
        if not configured:
            icon, note = "⚪", "_(chiave non configurata)_"
        elif result is not None:
            icon, note = "🟢", "operativo"
        else:
            icon, note = "🔴", "non raggiungibile"
        lines.append(f"{icon} *{name}* ({provider}) — {note}")

    lines.append("\n🎯 *Catena Master (ordine di priorità):*")
    for name, has_key in [
        ("Gemini 2.0 Flash", bool(GOOGLE_API_KEY)),
        ("Mistral Large",    True),
        ("DeepSeek V3",      True),
        ("Groq Llama 3.3",   True),
    ]:
        lines.append(f"{'✅' if has_key else '⚪'} {name}")

    try:
        await msg.edit_text("\n".join(lines), parse_mode="Markdown")
    except Exception:
        await msg.edit_text("\n".join(lines))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text.strip()
    if not user_query:
        return

    thinking_msg = await update.message.reply_text(
        "⚙️ *Consultando il Consiglio degli Esperti...*\n"
        "_Questo può richiedere fino a 45 secondi._",
        parse_mode="Markdown",
    )

    try:
        response = await run_ensemble_engine(user_query)
    except Exception as e:
        logger.exception(f"Errore imprevisto nel motore Ensemble: {e}")
        response = "❌ Si è verificato un errore interno. Riprova tra poco."

    await thinking_msg.delete()

    for chunk in _split_message(response, max_len=4000):
        try:
            await update.message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(chunk)

def _split_message(text: str, max_len: int = 4000) -> list[str]:
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
    return chunks or [text[:max_len]]


# ─────────────────────────────────────────────────────────────────
# [8] AVVIO APPLICAZIONE
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # FIX Python 3.14: crea esplicitamente il loop sul MainThread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    start_flask_thread()

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )

    application.add_handler(CommandHandler("start",  cmd_start))
    application.add_handler(CommandHandler("help",   cmd_help))
    application.add_handler(CommandHandler("ping",   cmd_ping))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Avvio del polling Telegram (Ensemble V4.1)...")
    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )

if __name__ == "__main__":
    main()
