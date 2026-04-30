async def call_gemini(self, client, prompt):
        """Versione Diagnostica: Legge il motivo esatto del fallimento"""
        if not self.keys['gemini']: return None
        
        # Proviamo il modello più generico possibile
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.keys['gemini'].strip()}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            resp = await client.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
            
            # QUI LA DIAGNOSI: Leggiamo cosa dice davvero Google
            error_details = resp.text
            logger.error(f"GEMINI FALLITO! Status: {resp.status_code} | Dettagli: {error_details}")
            
            # Se è un 404, proviamo l'ultima spiaggia: il modello 'gemini-pro' (vecchio ma stabile)
            if resp.status_code == 404:
                url_alt = url.replace("gemini-1.5-flash", "gemini-pro")
                resp_alt = await client.post(url_alt, json=payload, timeout=self.timeout)
                if resp_alt.status_code == 200:
                    return resp_alt.json()['candidates'][0]['content']['parts'][0]['text']
                    
            return None
        except Exception as e:
            logger.error(f"Errore connessione Gemini: {e}")
            return None
