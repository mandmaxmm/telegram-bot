"""
╔══════════════════════════════════════════════════════════════════╗
║        ENSEMBLE MULTI-MODEL ENGINE  — Architettura V4.2         ║
╠══════════════════════════════════════════════════════════════════╣
║  Novità V4.2:                                                    ║
║  • Gemini: catena interna di modelli (2.0-flash → 1.5-flash →   ║
║    2.0-flash-lite) — se uno fallisce prova il successivo         ║
║  • Cerebras aggiunto come Expert free ultra-rapido               ║
║  • OpenRouter: model ID aggiornati + set ampliato                ║
║  • /status testa anche i Maestri direttamente                    ║
║  • Solo TELEGRAM_TOKEN obbligatorio — tutto il resto opzionale   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, asyncio, logging, threading
from typing import Optional, Callable, Awaitable

import httpx
from flask import Flask
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
)

# ─────────────────────────────────────────────────────────────────
# [1] CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("EnsembleBot")

# Solo il token Telegram è strettamente obbligatorio
_REQUIRED = ["TELEGRAM_TOKEN"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    print(f"\n❌ Variabile obbligatoria mancante: TELEGRAM_TOKEN\n")
    raise SystemExit(1)

TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
GOOGLE_API_KEY     = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY    = os.environ.get("MISTRAL_API_KEY", "")
DEEPSEEK_API_KEY   = os.environ.get("DEEPSEEK_API_KEY", "")
XAI_API_KEY        = os.environ.get("XAI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_API_KEY   = os.environ.get("CEREBRAS_API_KEY", "")

PORT           = int(os.environ.get("PORT") or 10000)
EXPERT_TIMEOUT = 40
MASTER_TIMEOUT = 55

# Log riepilogativo delle chiavi al boot
_keys_status = {
    "GOOGLE_API_KEY":     bool(GOOGLE_API_KEY),
    "GROQ_API_KEY":       bool(GROQ_API_KEY),
    "MISTRAL_API_KEY":    bool(MISTRAL_API_KEY),
    "DEEPSEEK_API_KEY":   bool(DEEPSEEK_API_KEY),
    "XAI_API_KEY":        bool(XAI_API_KEY),
    "OPENROUTER_API_KEY": bool(OPENROUTER_API_KEY),
    "CEREBRAS_API_KEY":   bool(CEREBRAS_API_KEY),
}
for k, v in _keys_status.items():
    if v:
        logger.info(f"  ✅ {k} configurata")
    else:
        logger.warning(f"  ⚪ {k} non configurata — modelli correlati disabilitati")


# ─────────────────────────────────────────────────────────────────
# [2] FLASK
# ─────────────────────────────────────────────────────────────────

flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
@flask_app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "engine": "Ensemble V4.2"}, 200

def run_flask():
    import logging as _l; _l.getLogger("werkzeug").setLevel(_l.ERROR)
    flask_app.run(host="0.0.0.0", port=PORT)

def start_flask_thread():
    threading.Thread(target=run_flask, daemon=True, name="Flask").start()
    logger.info(f"Flask avviato sulla porta {PORT}")


# ─────────────────────────────────────────────────────────────────
# [3] HELPER ERRORI
# ─────────────────────────────────────────────────────────────────

_SILENT = {401: "chiave non valida", 402: "credito esaurito",
           403: "accesso negato",    404: "modello non trovato", 429: "rate limit"}

def _log_fail(label: str, exc: Exception) -> None:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _SILENT:
        logger.warning(f"[{label}] {_SILENT[exc.response.status_code]} — saltato")
    else:
        logger.warning(f"[{label}] {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────
# [4] EXPERT POOL
# ─────────────────────────────────────────────────────────────────

# ── Groq ────────────────────────────────────────────────────────
async def _call_groq(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not GROQ_API_KEY: return None
    try:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Groq", e); return None

# ── Cerebras (FREE — ultra-rapido, Llama 3.3 70B) ────────────────
async def _call_cerebras(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Cerebras offre inferenza gratuita su llama-3.3-70b.
    Chiave gratuita su: https://cloud.cerebras.ai
    """
    if not CEREBRAS_API_KEY: return None
    try:
        r = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
            json={"model": "llama-3.3-70b",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Cerebras", e); return None

# ── DeepSeek ─────────────────────────────────────────────────────
async def _call_deepseek(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not DEEPSEEK_API_KEY: return None
    try:
        r = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.6, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:DeepSeek", e); return None

# ── Mistral ──────────────────────────────────────────────────────
async def _call_mistral(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not MISTRAL_API_KEY: return None
    try:
        r = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": "mistral-large-latest",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.65, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Mistral", e); return None

# ── Grok ─────────────────────────────────────────────────────────
async def _call_grok(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not XAI_API_KEY: return None
    try:
        r = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {XAI_API_KEY}"},
            json={"model": "grok-3-latest",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.7, "max_tokens": 1024},
            timeout=EXPERT_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Grok", e); return None

# ── OpenRouter (hub free) ─────────────────────────────────────────
# I model ID free cambiano di frequente su OpenRouter.
# Usiamo una lista di ID alternativi per ogni "slot":
# se il primo è stato rimosso, il secondo prende il suo posto.

async def _call_openrouter_with_fallback(
    model_ids: list[str], label: str,
    prompt: str, client: httpx.AsyncClient
) -> Optional[str]:
    """Prova ogni model_id in sequenza fino al primo successo."""
    if not OPENROUTER_API_KEY: return None
    for mid in model_ids:
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://ensemble-bot.onrender.com",
                    "X-Title": "Ensemble Bot",
                },
                json={"model": mid,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.7, "max_tokens": 1024},
                timeout=EXPERT_TIMEOUT)
            r.raise_for_status()
            logger.info(f"[OR/{label}] Risposta da: {mid}")
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 429):
                logger.warning(f"[OR/{label}] {mid} → {_SILENT.get(e.response.status_code,'err')} — provo alternativa")
                continue
            _log_fail(f"Expert:OR/{label}", e); return None
        except Exception as e:
            _log_fail(f"Expert:OR/{label}", e); return None
    return None

# Gemini via OpenRouter — più ID alternativi in caso di rotazione
_OR_GEMINI_IDS = [
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-flash-1.5:free",
    "google/gemini-2.5-flash-preview:free",
]
_OR_LLAMA_IDS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "meta-llama/llama-3-70b-instruct:free",
]
_OR_DEEPSEEK_IDS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-chat:free",
    "deepseek/deepseek-r1:free",
]
_OR_MISC_IDS = [
    "qwen/qwen-2.5-72b-instruct:free",
    "qwen/qwq-32b:free",
    "microsoft/phi-4:free",
]

async def _call_or_gemini(p, c):
    return await _call_openrouter_with_fallback(_OR_GEMINI_IDS, "Gemini", p, c)
async def _call_or_llama(p, c):
    return await _call_openrouter_with_fallback(_OR_LLAMA_IDS, "Llama", p, c)
async def _call_or_deepseek(p, c):
    return await _call_openrouter_with_fallback(_OR_DEEPSEEK_IDS, "DeepSeek", p, c)
async def _call_or_misc(p, c):
    return await _call_openrouter_with_fallback(_OR_MISC_IDS, "Qwen/Phi", p, c)


async def gather_expert_opinions(prompt: str) -> dict[str, str]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _call_groq(prompt, client),
            _call_cerebras(prompt, client),
            _call_deepseek(prompt, client),
            _call_mistral(prompt, client),
            _call_grok(prompt, client),
            _call_or_gemini(prompt, client),
            _call_or_llama(prompt, client),
            _call_or_deepseek(prompt, client),
            _call_or_misc(prompt, client),
        )
    labels = [
        "Llama 3.3 70B (Groq)",
        "Llama 3.3 70B (Cerebras)",
        "DeepSeek V3",
        "Mistral Large",
        "Grok (xAI)",
        "Gemini (OpenRouter)",
        "Llama (OpenRouter)",
        "DeepSeek (OpenRouter)",
        "Qwen/Phi (OpenRouter)",
    ]
    opinions = {l: r for l, r in zip(labels, results) if r is not None}
    logger.info(f"Expert rispondenti ({len(opinions)}): {list(opinions.keys())}")
    return opinions


# ─────────────────────────────────────────────────────────────────
# [5] MASTER CHAIN
# ─────────────────────────────────────────────────────────────────
#
# GEMINI — catena di modelli interna:
#   gemini-2.0-flash → gemini-2.0-flash-lite → gemini-1.5-flash
#   Se un modello non esiste sull'account, il successivo viene provato
#   automaticamente senza toccare il codice.
# ─────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = (
    "Sei il Sintetizzatore di un sistema Ensemble Multi-Modello. "
    "Hai ricevuto la domanda originale dell'utente e i pareri indipendenti "
    "di piu modelli AI esperti. Il tuo compito:\n"
    "1. Analizza i pareri, identifica accordi e divergenze.\n"
    "2. Costruisci una risposta definitiva, autorevole e ben strutturata.\n"
    "3. Integra le prospettive complementari; ignora gli errori evidenti.\n"
    "4. NON menzionare i nomi dei modelli Expert ne i dettagli interni.\n"
    "5. Rispondi in italiano, in modo chiaro e completo."
)

def _build_synthesis_prompt(user_query: str, opinions: dict[str, str]) -> str:
    block = "\n\n".join(f"--- Parere {i+1} ---\n{t}" for i, t in enumerate(opinions.values()))
    return f"DOMANDA:\n{user_query}\n\nPARERI ESPERTI:\n{block}\n\nRisposta sintetica finale:"

# Modelli Gemini tentati in sequenza (dal più recente al più stabile)
_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

async def _master_gemini(sp: str) -> Optional[str]:
    """Tenta ogni modello Gemini fino al primo successo."""
    if not GOOGLE_API_KEY: return None
    for model in _GEMINI_MODELS:
        url = (f"https://generativelanguage.googleapis.com/v1/models/"
               f"{model}:generateContent?key={GOOGLE_API_KEY}")
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(url, json={
                    "system_instruction": {"parts": [{"text": SYNTHESIS_SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": sp}]}],
                    "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.4},
                }, timeout=MASTER_TIMEOUT)
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"[Master:Gemini] Risposta da: {model}")
                return text
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 404:
                logger.warning(f"[Master:Gemini] {model} non trovato — provo prossimo")
                continue  # prova il modello successivo
            if code in _SILENT:
                logger.warning(f"[Master:Gemini] {_SILENT[code]} — saltato")
                return None
            logger.warning(f"[Master:Gemini] HTTP {code} con {model}")
            continue
        except Exception as e:
            _log_fail(f"Master:Gemini/{model}", e)
            continue
    logger.warning("[Master:Gemini] Tutti i modelli Gemini falliti.")
    return None

async def _master_mistral(sp: str) -> Optional[str]:
    if not MISTRAL_API_KEY: return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                json={"model": "mistral-large-latest",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user", "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:Mistral", e); return None

async def _master_deepseek(sp: str) -> Optional[str]:
    if not DEEPSEEK_API_KEY: return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={"model": "deepseek-chat",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user", "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:DeepSeek", e); return None

async def _master_groq(sp: str) -> Optional[str]:
    if not GROQ_API_KEY: return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.3-70b-versatile",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user", "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:Groq", e); return None

async def _master_cerebras(sp: str) -> Optional[str]:
    if not CEREBRAS_API_KEY: return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                json={"model": "llama-3.3-70b",
                      "messages": [{"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                                    {"role": "user", "content": sp}],
                      "temperature": 0.4, "max_tokens": 2048},
                timeout=MASTER_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:Cerebras", e); return None

MASTER_CHAIN: list[tuple[str, Callable[[str], Awaitable[Optional[str]]]]] = [
    ("Gemini",   _master_gemini),
    ("Mistral",  _master_mistral),
    ("DeepSeek", _master_deepseek),
    ("Cerebras", _master_cerebras),
    ("Groq",     _master_groq),
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
    return max(opinions.values(), key=lambda t: len(t.split()))

async def run_ensemble_engine(user_query: str) -> str:
    opinions = await gather_expert_opinions(user_query)
    if not opinions:
        return ("⚠️ Nessun modello Expert ha risposto.\n"
                "Controlla /status per vedere quali chiavi API sono attive.")
    sp = _build_synthesis_prompt(user_query, opinions)
    final_response, _ = await run_master_chain(sp)
    if final_response is None:
        logger.error("Tutta la Master Chain fallita — uso miglior risposta Expert.")
        final_response = _best_single_expert(opinions)
    return final_response


# ─────────────────────────────────────────────────────────────────
# [7] HANDLER TELEGRAM
# ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    active = []
    if GROQ_API_KEY:       active.append("Groq (Llama 3.3)")
    if CEREBRAS_API_KEY:   active.append("Cerebras (Llama 3.3)")
    if DEEPSEEK_API_KEY:   active.append("DeepSeek V3")
    if MISTRAL_API_KEY:    active.append("Mistral Large")
    if XAI_API_KEY:        active.append("Grok")
    if OPENROUTER_API_KEY: active.append("OpenRouter (4 modelli free)")
    experts_str = ", ".join(active) if active else "nessuno configurato — aggiungi le chiavi API"
    text = (
        f"👋 *Benvenuto nel Motore Ensemble V4.2!*\n\n"
        f"🤖 *Expert attivi:* {experts_str}\n"
        f"🎯 *Master Chain:* Gemini → Mistral → DeepSeek → Cerebras → Groq\n\n"
        f"Usa /status per vedere lo stato live di tutti i modelli.\n"
        f"Scrivi qualsiasi domanda per iniziare!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *Comandi:*\n\n"
        "/start  — Panoramica\n"
        "/help   — Questo messaggio\n"
        "/ping   — Health check\n"
        "/status — Stato live modelli (Expert + Master)",
        parse_mode="Markdown"
    )

async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🟢 Pong! Ensemble V4.2 operativo.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Testa Expert con probe leggero + verifica la Master Chain
    tentando una sintesi su testo minimo.
    """
    probe = "Rispondi solo con la parola OK."
    msg = await update.message.reply_text("🔍 Controllo stato modelli in corso...")

    # Test Expert
    async with httpx.AsyncClient() as client:
        expert_results = await asyncio.gather(
            _call_groq(probe, client),
            _call_cerebras(probe, client),
            _call_deepseek(probe, client),
            _call_mistral(probe, client),
            _call_grok(probe, client),
            _call_or_gemini(probe, client),
            _call_or_llama(probe, client),
            _call_or_deepseek(probe, client),
            _call_or_misc(probe, client),
        )

    expert_rows = [
        ("Llama 3.3 70B",    "Groq",       bool(GROQ_API_KEY)),
        ("Llama 3.3 70B",    "Cerebras",   bool(CEREBRAS_API_KEY)),
        ("DeepSeek V3",      "DeepSeek",   bool(DEEPSEEK_API_KEY)),
        ("Mistral Large",    "Mistral",    bool(MISTRAL_API_KEY)),
        ("Grok",             "xAI",        bool(XAI_API_KEY)),
        ("Gemini (free)",    "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Llama (free)",     "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("DeepSeek (free)",  "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Qwen/Phi (free)",  "OpenRouter", bool(OPENROUTER_API_KEY)),
    ]

    # Test Master Chain direttamente
    test_sp = _build_synthesis_prompt("Test di sistema", {"Test": "Il sistema funziona."})
    master_results = {}
    for name, fn in MASTER_CHAIN:
        res = await fn(test_sp)
        master_results[name] = res is not None

    lines = ["🤖 *Expert:*\n"]
    for (name, provider, configured), result in zip(expert_rows, expert_results):
        if not configured:
            icon, note = "⚪", "_(chiave mancante)_"
        elif result is not None:
            icon, note = "🟢", "operativo"
        else:
            icon, note = "🔴", "non raggiungibile"
        lines.append(f"{icon} *{name}* ({provider}) — {note}")

    lines.append("\n🎯 *Master Chain:*\n")
    for name, ok in master_results.items():
        icon = "🟢" if ok else ("🔴" if any([
            name == "Gemini"   and GOOGLE_API_KEY,
            name == "Mistral"  and MISTRAL_API_KEY,
            name == "DeepSeek" and DEEPSEEK_API_KEY,
            name == "Cerebras" and CEREBRAS_API_KEY,
            name == "Groq"     and GROQ_API_KEY,
        ]) else "⚪")
        lines.append(f"{icon} {name}")

    try:
        await msg.edit_text("\n".join(lines), parse_mode="Markdown")
    except Exception:
        await msg.edit_text("\n".join(lines))

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_query = update.message.text.strip()
    if not user_query: return

    thinking_msg = await update.message.reply_text(
        "⚙️ *Consultando il Consiglio degli Esperti...*\n"
        "_Questo può richiedere fino a 45 secondi._",
        parse_mode="Markdown",
    )
    try:
        response = await run_ensemble_engine(user_query)
    except Exception as e:
        logger.exception(f"Errore nel motore: {e}")
        response = "❌ Errore interno. Riprova tra poco."

    await thinking_msg.delete()
    for chunk in _split_message(response):
        try:
            await update.message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(chunk)

def _split_message(text: str, max_len: int = 4000) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_len:
            if current: chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para) if current else para
    if current: chunks.append(current.strip())
    return chunks or [text[:max_len]]


# ─────────────────────────────────────────────────────────────────
# [8] AVVIO
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_flask_thread()

    app = (Application.builder().token(TELEGRAM_TOKEN)
           .connect_timeout(30).read_timeout(30).write_timeout(30).build())

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("ping",   cmd_ping))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Avvio polling Telegram (Ensemble V4.2)...")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
