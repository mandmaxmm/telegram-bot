"""
╔══════════════════════════════════════════════════════════════════╗
║        ENSEMBLE MULTI-MODEL ENGINE  — Architettura V4.3         ║
║        Telegram Bot · Deploy: Render · Backend: Python          ║
╠══════════════════════════════════════════════════════════════════╣
║  Novità V4.3:                                                    ║
║  • NVIDIA NIM: 3 Expert (Llama 3.3, DeepSeek R1, Gemma 3 27B)   ║
║  • NVIDIA NIM come Maestro nella catena di sintesi               ║
║  • Gemini: catena interna 4 modelli (404-proof)                  ║
║  • OpenRouter: lista ID alternativi per ogni slot (rotation-safe)║
║  • Cerebras: Expert + Maestro free ultra-rapido                  ║
║  • /status testa Expert e Master Chain live                      ║
║  • Tutti i provider opzionali — solo TELEGRAM_TOKEN obbligatorio ║
╚══════════════════════════════════════════════════════════════════╝

SEZIONI:
  [1] CONFIGURAZIONE & COSTANTI
  [2] MICRO-SERVER FLASK      (Port-binding Render)
  [3] HELPER ERRORI HTTP      (Classificazione silenziosa)
  [4] EXPERT POOL             (Chiamate parallele)
  [5] MASTER CHAIN            (Sintetizzatori a cascata)
  [6] PIPELINE ENSEMBLE       (Orchestrazione)
  [7] HANDLER TELEGRAM        (Comandi e messaggi)
  [8] AVVIO APPLICAZIONE
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

# Solo TELEGRAM_TOKEN è strettamente obbligatorio.
# Tutti gli altri provider sono opzionali: se la chiave manca
# il modello viene semplicemente saltato in silenzio.
_REQUIRED = ["TELEGRAM_TOKEN"]
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    print("\n❌ TELEGRAM_TOKEN mancante — impossibile avviare il bot.\n")
    raise SystemExit(1)

TELEGRAM_TOKEN     = os.environ["TELEGRAM_TOKEN"]
GOOGLE_API_KEY     = os.environ.get("GOOGLE_API_KEY",     "")
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY",       "")
MISTRAL_API_KEY    = os.environ.get("MISTRAL_API_KEY",    "")
DEEPSEEK_API_KEY   = os.environ.get("DEEPSEEK_API_KEY",   "")
XAI_API_KEY        = os.environ.get("XAI_API_KEY",        "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_API_KEY   = os.environ.get("CEREBRAS_API_KEY",   "")
NVIDIA_API_KEY     = os.environ.get("NVIDIA_API_KEY",     "")

PORT           = int(os.environ.get("PORT") or 10000)
EXPERT_TIMEOUT = 40   # secondi per ogni Expert
MASTER_TIMEOUT = 55   # secondi per ogni tentativo Master

# Riepilogo chiavi al boot — utile per debug su Render
_ALL_KEYS = {
    "GOOGLE_API_KEY":     GOOGLE_API_KEY,
    "GROQ_API_KEY":       GROQ_API_KEY,
    "MISTRAL_API_KEY":    MISTRAL_API_KEY,
    "DEEPSEEK_API_KEY":   DEEPSEEK_API_KEY,
    "XAI_API_KEY":        XAI_API_KEY,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "CEREBRAS_API_KEY":   CEREBRAS_API_KEY,
    "NVIDIA_API_KEY":     NVIDIA_API_KEY,
}
for _k, _v in _ALL_KEYS.items():
    if _v:
        logger.info(f"  ✅ {_k} configurata")
    else:
        logger.warning(f"  ⚪ {_k} non configurata — provider disabilitato")


# ─────────────────────────────────────────────────────────────────
# [2] MICRO-SERVER FLASK  ·  Port-binding per Render
# ─────────────────────────────────────────────────────────────────
# Render richiede che un Web Service apra una porta HTTP.
# Flask gira su un thread daemon separato e risponde 200 su /health.

flask_app = Flask(__name__)

@flask_app.route("/", methods=["GET"])
@flask_app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok", "engine": "Ensemble V4.3"}, 200

def run_flask() -> None:
    import logging as _l
    _l.getLogger("werkzeug").setLevel(_l.ERROR)
    flask_app.run(host="0.0.0.0", port=PORT)

def start_flask_thread() -> None:
    threading.Thread(target=run_flask, daemon=True, name="Flask").start()
    logger.info(f"Flask health-server avviato sulla porta {PORT}")


# ─────────────────────────────────────────────────────────────────
# [3] HELPER ERRORI HTTP  ·  Classificazione silenziosa
# ─────────────────────────────────────────────────────────────────
# Un modello con credito esaurito o chiave scaduta non è un errore
# del sistema. I codici 4xx vengono loggati a WARNING e il modello
# viene saltato silenziosamente senza interrompere il flusso.

_SILENT_CODES = {
    401: "chiave non valida",
    402: "credito esaurito",
    403: "accesso negato",
    404: "modello non trovato",
    429: "rate limit",
}

def _log_fail(label: str, exc: Exception) -> None:
    """Logga il fallimento di un modello al livello appropriato."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        reason = _SILENT_CODES.get(code, f"HTTP {code}")
        logger.warning(f"[{label}] Non disponibile ({reason}) — saltato")
    else:
        logger.warning(f"[{label}] {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────────────────────────
# [4] EXPERT POOL  ·  Chiamate parallele ai provider
# ─────────────────────────────────────────────────────────────────

# ── Groq (FREE — Llama 3.3 70B, velocità estrema) ────────────────
# Chiave gratuita: https://console.groq.com
async def _call_groq(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        r = await client.post(
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
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Groq/Llama3.3", e)
        return None


# ── Cerebras (FREE — Llama 3.3 70B, inferenza ultra-rapida) ──────
# Chiave gratuita: https://cloud.cerebras.ai
async def _call_cerebras(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not CEREBRAS_API_KEY:
        return None
    try:
        r = await client.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
            json={
                "model": "llama-3.3-70b",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Cerebras/Llama3.3", e)
        return None


# ── DeepSeek V3 (logica e matematica) ────────────────────────────
# Chiave: https://platform.deepseek.com
async def _call_deepseek(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        return None
    try:
        r = await client.post(
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
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:DeepSeek", e)
        return None


# ── Mistral Large (ragionamento strutturato) ──────────────────────
# Chiave: https://console.mistral.ai
async def _call_mistral(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not MISTRAL_API_KEY:
        return None
    try:
        r = await client.post(
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
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Mistral", e)
        return None


# ── Grok xAI (dati real-time, prospettive alternative) ───────────
# Chiave: https://console.x.ai
async def _call_grok(prompt: str, client: httpx.AsyncClient) -> Optional[str]:
    if not XAI_API_KEY:
        return None
    try:
        r = await client.post(
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
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Expert:Grok", e)
        return None


# ── NVIDIA NIM (crediti free all'iscrizione) ──────────────────────
# Endpoint OpenAI-compatibile. Chiave: https://build.nvidia.com
# Modelli disponibili: meta/llama-3.3-70b-instruct,
#   deepseek-ai/deepseek-r1, google/gemma-3-27b-it

async def _call_nvidia(
    model_id: str, label: str,
    prompt: str, client: httpx.AsyncClient,
) -> Optional[str]:
    """Helper generico per qualsiasi modello via NVIDIA NIM."""
    if not NVIDIA_API_KEY:
        return None
    try:
        r = await client.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
            timeout=EXPERT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail(f"Expert:NVIDIA/{label}", e)
        return None

async def _call_nvidia_llama(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_nvidia("meta/llama-3.3-70b-instruct", "Llama3.3", p, c)

async def _call_nvidia_deepseek_r1(p: str, c: httpx.AsyncClient) -> Optional[str]:
    """DeepSeek R1 — ragionamento chain-of-thought: arricchisce il materiale del Maestro."""
    return await _call_nvidia("deepseek-ai/deepseek-r1", "DeepSeek-R1", p, c)

async def _call_nvidia_gemma(p: str, c: httpx.AsyncClient) -> Optional[str]:
    """Gemma 3 27B — prospettiva Google, diversità nel pool."""
    return await _call_nvidia("google/gemma-3-27b-it", "Gemma3-27B", p, c)


# ── OpenRouter (hub multi-modello, tier free) ─────────────────────
# I model ID free ruotano spesso: ogni slot ha una lista di
# ID alternativi — se il primo restituisce 404 si prova il successivo.
# Chiave gratuita: https://openrouter.ai/keys

async def _call_openrouter_with_fallback(
    model_ids: list[str],
    label: str,
    prompt: str,
    client: httpx.AsyncClient,
) -> Optional[str]:
    """Prova ogni model_id in sequenza fino al primo successo."""
    if not OPENROUTER_API_KEY:
        return None
    for mid in model_ids:
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://ensemble-bot.onrender.com",
                    "X-Title": "Ensemble Multi-Model Bot",
                },
                json={
                    "model": mid,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=EXPERT_TIMEOUT,
            )
            r.raise_for_status()
            logger.info(f"[OR/{label}] Risposta da: {mid}")
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 429):
                logger.warning(f"[OR/{label}] {mid} — {_SILENT_CODES.get(e.response.status_code,'err')}, provo alternativa")
                continue
            _log_fail(f"Expert:OR/{label}", e)
            return None
        except Exception as e:
            _log_fail(f"Expert:OR/{label}", e)
            return None
    return None

# Liste di ID alternativi per ogni slot OpenRouter
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
    "mistralai/mistral-7b-instruct:free",
]

async def _call_or_gemini(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter_with_fallback(_OR_GEMINI_IDS, "Gemini", p, c)

async def _call_or_llama(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter_with_fallback(_OR_LLAMA_IDS, "Llama", p, c)

async def _call_or_deepseek(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter_with_fallback(_OR_DEEPSEEK_IDS, "DeepSeek", p, c)

async def _call_or_misc(p: str, c: httpx.AsyncClient) -> Optional[str]:
    return await _call_openrouter_with_fallback(_OR_MISC_IDS, "Qwen/Phi", p, c)


# ── Raccolta parallela di tutti gli Expert ────────────────────────

async def gather_expert_opinions(prompt: str) -> dict[str, str]:
    """
    Interroga tutti gli Expert simultaneamente con asyncio.gather.
    Tempo totale = tempo del modello più lento tra i sopravvissuti.
    Restituisce {nome: risposta} filtrando i None.
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            # Provider diretti
            _call_groq(prompt, client),
            _call_cerebras(prompt, client),
            _call_deepseek(prompt, client),
            _call_mistral(prompt, client),
            _call_grok(prompt, client),
            # NVIDIA NIM
            _call_nvidia_llama(prompt, client),
            _call_nvidia_deepseek_r1(prompt, client),
            _call_nvidia_gemma(prompt, client),
            # OpenRouter free
            _call_or_gemini(prompt, client),
            _call_or_llama(prompt, client),
            _call_or_deepseek(prompt, client),
            _call_or_misc(prompt, client),
        )

    labels = [
        "Llama 3.3 70B (Groq)",
        "Llama 3.3 70B (Cerebras)",
        "DeepSeek V3 (DeepSeek)",
        "Mistral Large (Mistral)",
        "Grok 3 (xAI)",
        "Llama 3.3 70B (NVIDIA NIM)",
        "DeepSeek R1 (NVIDIA NIM)",
        "Gemma 3 27B (NVIDIA NIM)",
        "Gemini (OpenRouter free)",
        "Llama (OpenRouter free)",
        "DeepSeek (OpenRouter free)",
        "Qwen/Phi (OpenRouter free)",
    ]

    opinions = {
        label: result
        for label, result in zip(labels, results)
        if result is not None
    }
    logger.info(f"Expert rispondenti ({len(opinions)}): {list(opinions.keys())}")
    return opinions


# ─────────────────────────────────────────────────────────────────
# [5] MASTER CHAIN  ·  Sintetizzatori a cascata
# ─────────────────────────────────────────────────────────────────
#
# Ordine di priorità:
#   1. Gemini 2.0 Flash  (catena interna 4 modelli — 404-proof)
#   2. Mistral Large
#   3. DeepSeek V3
#   4. NVIDIA NIM        (Llama 3.3 — crediti free)
#   5. Cerebras          (Llama 3.3 — free, ultra-rapido)
#   6. Groq              (anchor — sempre disponibile)
#
# Al primo successo la catena si ferma.
# Se tutti falliscono → miglior risposta Expert singola (no dump grezzi).
# ─────────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = (
    "Sei il Sintetizzatore di un sistema Ensemble Multi-Modello. "
    "Hai ricevuto la domanda originale dell'utente e i pareri indipendenti "
    "di piu modelli AI esperti. Il tuo compito:\n"
    "1. Analizza i pareri, identifica accordi e divergenze.\n"
    "2. Costruisci una risposta definitiva, autorevole e ben strutturata.\n"
    "3. Integra le prospettive complementari; ignora errori evidenti.\n"
    "4. NON menzionare nomi di modelli ne dettagli del processo interno.\n"
    "5. Rispondi in italiano, direttamente all'utente, in modo chiaro e completo."
)

def _build_synthesis_prompt(user_query: str, opinions: dict[str, str]) -> str:
    """Costruisce il mega-prompt per il Modello Maestro."""
    block = "\n\n".join(
        f"--- Parere {i + 1} ---\n{text}"
        for i, text in enumerate(opinions.values())
    )
    return (
        f"DOMANDA ORIGINALE:\n{user_query}\n\n"
        f"PARERI DEGLI ESPERTI:\n{block}\n\n"
        "Produci ora la risposta sintetica finale."
    )


# Maestro 1 — Gemini con catena interna di modelli (404-proof)
_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

async def _master_gemini(sp: str) -> Optional[str]:
    """
    Tenta ogni modello Gemini in sequenza.
    Un 404 significa 'modello non disponibile su questo account':
    si prova il successivo senza interrompere il flusso.
    """
    if not GOOGLE_API_KEY:
        return None
    for model in _GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1/models/"
            f"{model}:generateContent?key={GOOGLE_API_KEY}"
        )
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(
                    url,
                    json={
                        "system_instruction": {"parts": [{"text": SYNTHESIS_SYSTEM_PROMPT}]},
                        "contents": [{"parts": [{"text": sp}]}],
                        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.4},
                    },
                    timeout=MASTER_TIMEOUT,
                )
                r.raise_for_status()
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"[Master:Gemini] Risposta ottenuta da: {model}")
                return text
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 404:
                logger.warning(f"[Master:Gemini] {model} non trovato — provo prossimo modello")
                continue
            logger.warning(f"[Master:Gemini] {model} → {_SILENT_CODES.get(code, f'HTTP {code}')} — saltato")
            return None
        except Exception as e:
            _log_fail(f"Master:Gemini/{model}", e)
            continue
    logger.warning("[Master:Gemini] Tutti i modelli Gemini non disponibili.")
    return None


# Maestro 2 — Mistral Large
async def _master_mistral(sp: str) -> Optional[str]:
    if not MISTRAL_API_KEY:
        return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                json={
                    "model": "mistral-large-latest",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": sp},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2048,
                },
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:Mistral", e)
        return None


# Maestro 3 — DeepSeek V3
async def _master_deepseek(sp: str) -> Optional[str]:
    if not DEEPSEEK_API_KEY:
        return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": sp},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2048,
                },
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:DeepSeek", e)
        return None


# Maestro 4 — NVIDIA NIM (crediti free, Llama 3.3 70B)
async def _master_nvidia(sp: str) -> Optional[str]:
    if not NVIDIA_API_KEY:
        return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
                json={
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": sp},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2048,
                },
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:NVIDIA", e)
        return None


# Maestro 5 — Cerebras (free, ultra-rapido)
async def _master_cerebras(sp: str) -> Optional[str]:
    if not CEREBRAS_API_KEY:
        return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                json={
                    "model": "llama-3.3-70b",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": sp},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2048,
                },
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:Cerebras", e)
        return None


# Maestro 6 — Groq (anchor, sempre disponibile se chiave presente)
async def _master_groq(sp: str) -> Optional[str]:
    if not GROQ_API_KEY:
        return None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": sp},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2048,
                },
                timeout=MASTER_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        _log_fail("Master:Groq", e)
        return None


# Catena ordinata: primo che risponde vince
MASTER_CHAIN: list[tuple[str, Callable[[str], Awaitable[Optional[str]]]]] = [
    ("Gemini",   _master_gemini),
    ("Mistral",  _master_mistral),
    ("DeepSeek", _master_deepseek),
    ("NVIDIA",   _master_nvidia),
    ("Cerebras", _master_cerebras),
    ("Groq",     _master_groq),
]

async def run_master_chain(sp: str) -> tuple[Optional[str], Optional[str]]:
    """Scorre la catena fino al primo successo. Restituisce (risposta, nome_maestro)."""
    for name, fn in MASTER_CHAIN:
        result = await fn(sp)
        if result:
            logger.info(f"Sintesi completata da Master: {name}")
            return result, name
    return None, None


# ─────────────────────────────────────────────────────────────────
# [6] PIPELINE ENSEMBLE  ·  Orchestrazione completa
# ─────────────────────────────────────────────────────────────────

def _best_single_expert(opinions: dict[str, str]) -> str:
    """
    Fallback finale: seleziona la risposta Expert più completa.
    Proxy di qualità: numero di parole (la più lunga = più dettagliata).
    NON produce dump multipli — restituisce UNA sola risposta pulita.
    """
    return max(opinions.values(), key=lambda t: len(t.split()))

async def run_ensemble_engine(user_query: str) -> str:
    """
    Pipeline principale — completamente disaccoppiata da Telegram.
    Pronta per essere esposta come endpoint REST in Fase 2.

    Flusso:
      1. Interrogazione parallela di tutti gli Expert
      2. Costruzione del prompt di sintesi
      3. Master Chain: Gemini → Mistral → DeepSeek → NVIDIA → Cerebras → Groq
      4. Fallback finale: miglior risposta Expert singola
    """
    opinions = await gather_expert_opinions(user_query)

    if not opinions:
        return (
            "⚠️ Nessun modello Expert ha risposto in questo momento.\n"
            "Usa /status per verificare lo stato dei provider "
            "e controlla le chiavi API su Render."
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
    active = []
    if GROQ_API_KEY:       active.append("Groq")
    if CEREBRAS_API_KEY:   active.append("Cerebras")
    if DEEPSEEK_API_KEY:   active.append("DeepSeek")
    if MISTRAL_API_KEY:    active.append("Mistral")
    if XAI_API_KEY:        active.append("Grok")
    if NVIDIA_API_KEY:     active.append("NVIDIA NIM")
    if OPENROUTER_API_KEY: active.append("OpenRouter")
    experts_str = ", ".join(active) if active else "nessuno — aggiungi le chiavi su Render"
    text = (
        "👋 *Benvenuto nel Motore Ensemble V4.3!*\n\n"
        "Sono un sistema di intelligenza collettiva che interroga "
        "simultaneamente fino a 12 modelli AI e fonde le loro risposte "
        "in un'unica risposta autorevole.\n\n"
        f"🤖 *Provider attivi:* {experts_str}\n"
        "🎯 *Master Chain:* Gemini → Mistral → DeepSeek → NVIDIA → Cerebras → Groq\n\n"
        "Usa /status per vedere lo stato live di ogni modello.\n"
        "Scrivi qualsiasi domanda per iniziare!"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *Comandi disponibili:*\n\n"
        "/start  — Panoramica del sistema\n"
        "/help   — Questo messaggio\n"
        "/ping   — Health check rapido\n"
        "/status — Stato live Expert + Master Chain",
        parse_mode="Markdown",
    )


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🟢 Pong! Ensemble V4.3 operativo.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Testa ogni modello Expert con un probe leggero,
    poi verifica ogni Maestro con una sintesi minima.
    """
    probe_expert = "Rispondi solo con la parola OK."
    probe_master = _build_synthesis_prompt(
        "Test di sistema",
        {"Test": "Il sistema funziona correttamente."}
    )

    msg = await update.message.reply_text("🔍 Controllo stato modelli in corso...")

    # Test Expert
    async with httpx.AsyncClient() as client:
        expert_results = await asyncio.gather(
            _call_groq(probe_expert, client),
            _call_cerebras(probe_expert, client),
            _call_deepseek(probe_expert, client),
            _call_mistral(probe_expert, client),
            _call_grok(probe_expert, client),
            _call_nvidia_llama(probe_expert, client),
            _call_nvidia_deepseek_r1(probe_expert, client),
            _call_nvidia_gemma(probe_expert, client),
            _call_or_gemini(probe_expert, client),
            _call_or_llama(probe_expert, client),
            _call_or_deepseek(probe_expert, client),
            _call_or_misc(probe_expert, client),
        )

    expert_rows = [
        ("Llama 3.3 70B",    "Groq",       bool(GROQ_API_KEY)),
        ("Llama 3.3 70B",    "Cerebras",   bool(CEREBRAS_API_KEY)),
        ("DeepSeek V3",      "DeepSeek",   bool(DEEPSEEK_API_KEY)),
        ("Mistral Large",    "Mistral",    bool(MISTRAL_API_KEY)),
        ("Grok 3",           "xAI",        bool(XAI_API_KEY)),
        ("Llama 3.3 70B",    "NVIDIA NIM", bool(NVIDIA_API_KEY)),
        ("DeepSeek R1",      "NVIDIA NIM", bool(NVIDIA_API_KEY)),
        ("Gemma 3 27B",      "NVIDIA NIM", bool(NVIDIA_API_KEY)),
        ("Gemini (free)",    "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Llama (free)",     "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("DeepSeek (free)",  "OpenRouter", bool(OPENROUTER_API_KEY)),
        ("Qwen/Phi (free)",  "OpenRouter", bool(OPENROUTER_API_KEY)),
    ]

    # Test Master
    master_results = {}
    for name, fn in MASTER_CHAIN:
        res = await fn(probe_master)
        master_results[name] = res is not None

    # Costruzione risposta
    lines = ["🤖 *Expert Pool:*\n"]
    for (name, provider, configured), result in zip(expert_rows, expert_results):
        if not configured:
            icon, note = "⚪", "_(chiave non configurata)_"
        elif result is not None:
            icon, note = "🟢", "operativo"
        else:
            icon, note = "🔴", "non raggiungibile"
        lines.append(f"{icon} *{name}* ({provider}) — {note}")

    lines.append("\n🎯 *Master Chain:*\n")
    master_has_key = {
        "Gemini":   bool(GOOGLE_API_KEY),
        "Mistral":  bool(MISTRAL_API_KEY),
        "DeepSeek": bool(DEEPSEEK_API_KEY),
        "NVIDIA":   bool(NVIDIA_API_KEY),
        "Cerebras": bool(CEREBRAS_API_KEY),
        "Groq":     bool(GROQ_API_KEY),
    }
    for name, ok in master_results.items():
        if not master_has_key.get(name, False):
            icon = "⚪"
        elif ok:
            icon = "🟢"
        else:
            icon = "🔴"
        lines.append(f"{icon} {name}")

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
    """Divide testo lungo in chunks rispettando i paragrafi."""
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
    # FIX Python 3.14 — crea esplicitamente il loop sul MainThread
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

    logger.info("Avvio polling Telegram (Ensemble V4.3)...")
    application.run_polling(
        drop_pending_updates=True,   # risolve zombie instances al riavvio Render
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
