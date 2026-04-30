# ... (Parti invariate di Flask e setup) ...

class EnsembleEngine:
    def __init__(self):
        self.timeout = httpx.Timeout(20.0, connect=5.0)
        self.keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "mistral": os.getenv("MISTRAL_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "xai": os.getenv("XAI_API_KEY")
        }

    async def call_api(self, client, url, key, model, prompt, provider):
        if not key: return ""
        try:
            headers = {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}
            if provider == "OpenRouter":
                headers.update({"HTTP-Referer": "https://render.com", "X-Title": "EnsembleBot"})
            
            # Nota: Alcuni provider richiedono parametri specifici, teniamo il payload standard OpenAI
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
            resp = await client.post(url, json=payload, headers=headers, timeout=self.timeout)
            
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content']
            
            logger.error(f"Errore {provider} ({model}): {resp.status_code} - {resp.text}")
            return ""
        except Exception as e:
            logger.error(f"Ex {provider}: {e}")
            return ""

    async def call_gemini(self, client, prompt):
        if not self.keys['gemini']: return ""
        try:
            # Passiamo alla V1 stabile
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            logger.error(f"Errore Gemini V1: {resp.status_code}")
            return ""
        except Exception as e:
            logger.error(f"Ex Gemini: {e}")
            return ""

    async def get_response(self, query):
        async with httpx.AsyncClient() as client:
            # Pool Esperti (Usiamo modelli di cui siamo certi dell'esistenza)
            tasks = [
                self.call_api(client, "https://api.groq.com/openai/v1/chat/completions", self.keys['groq'], "llama-3.3-70b-versatile", query, "Groq"),
                self.call_api(client, "https://api.mistral.ai/v1/chat/completions", self.keys['mistral'], "mistral-large-latest", query, "Mistral"),
                self.call_api(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "google/gemini-2.0-flash-001", query, "OpenRouter")
            ]
            
            resps = await asyncio.gather(*tasks)
            valid = [r for r in resps if r]
            
            if not valid: return "❌ Nessun esperto ha risposto (verifica Groq/Mistral/OpenRouter)."

            context = "\n\n".join([f"Fonte {i+1}: {r}" for i, r in enumerate(valid)])
            summary_prompt = f"Sei un assistente brillante. Riassumi queste risposte in modo naturale:\n\n{context}"

            # Strategia di Sintesi: Gemini V1 -> OpenRouter -> Fallback Llama/Mistral
            final_synthesis = await self.call_gemini(client, summary_prompt)
            
            if not final_synthesis:
                final_synthesis = await self.call_api(client, "https://openrouter.ai/api/v1/chat/completions", self.keys['openrouter'], "google/gemini-2.0-flash-001", summary_prompt, "Sintesi-OR")
            
            return final_synthesis if final_synthesis else f"[Sintesi Rapida]\n{valid[0]}"

# ... (Resto del codice handler e main uguale) ...
