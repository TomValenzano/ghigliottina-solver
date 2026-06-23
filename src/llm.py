"""Motore LLM intercambiabile: Gemini (free tier) oppure Ollama (locale).

Tutti i motori espongono lo stesso metodo:
    complete(prompt, system=None, temperature=0.0, max_tokens=512) -> str

`get_engine(name)` restituisce l'istanza giusta in base al nome
("gemini", "ollama", "mock").
"""
from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod

import config


class LLMEngine(ABC):
    name: str = "base"

    @abstractmethod
    def complete(
        self, prompt: str, system: str | None = None,
        temperature: float = 0.0, max_tokens: int = 512,
    ) -> str: ...


class GeminiEngine(LLMEngine):
    """Usa il nuovo SDK google-genai (pip install google-genai).

    Tiene attivo il 'thinking' dei modelli 2.5 (migliora la scelta della
    soluzione) ma gestisce le risposte vuote/troncate: se l'output è vuoto
    (ragionamento che esaurisce il budget) ritenta con più token. Gestisce
    anche i rate limit del free tier con backoff.
    """
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from google import genai
        from google.genai import types

        key = api_key or config.GEMINI_API_KEY
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY non impostata. Crea una key gratuita su "
                "https://aistudio.google.com ed esportala."
            )
        self.client = genai.Client(api_key=key)
        self.model_name = model or config.GEMINI_MODEL
        self._types = types

    def _generate(self, prompt, system, temperature, max_tokens):
        import time

        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            # thinking nativo disattivato: ragionamento fatto in modo esplicito
            # nel prompt; evita risposte vuote/troncate e raddoppio chiamate
            thinking_config=self._types.ThinkingConfig(thinking_budget=0),
        )
        # backoff breve e limitato per i rate limit del free tier
        for attempt, wait in enumerate((5, 10)):
            try:
                return self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=cfg
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "quota" in msg:
                    time.sleep(wait)
                    continue
                raise
        # ultimo tentativo senza catturare l'eccezione
        return self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=cfg
        )

    @staticmethod
    def _extract_text(resp) -> str:
        if resp is None:
            return ""
        try:
            if resp.text:
                return resp.text
        except Exception:  # noqa: BLE001
            pass
        out: list[str] = []
        for c in getattr(resp, "candidates", None) or []:
            parts = getattr(getattr(c, "content", None), "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    out.append(t)
        return "".join(out)

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        resp = self._generate(prompt, system, temperature, max_tokens)
        return self._extract_text(resp).strip()


class OllamaEngine(LLMEngine):
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or config.OLLAMA_MODEL
        self.host = (host or config.OLLAMA_HOST).rstrip("/")

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (data.get("response") or "").strip()


class MockEngine(LLMEngine):
    """Per test offline: ritorna il primo candidato / una descrizione fissa."""
    name = "mock"

    def complete(self, prompt, system=None, temperature=0.0, max_tokens=512):
        # fase [C]: il system della scelta soluzione richiede output JSON
        if "choice" in (system or "").lower():
            return 'Ragionamento di prova.\n{"choice": 1}'
        return "Descrizione di prova generata dal motore mock."


def get_engine(name: str) -> LLMEngine:
    name = (name or "").lower()
    if name == "gemini":
        return GeminiEngine()
    if name == "ollama":
        return OllamaEngine()
    if name == "mock":
        return MockEngine()
    raise ValueError(f"Motore LLM sconosciuto: {name!r}")
