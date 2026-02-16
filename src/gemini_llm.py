from __future__ import annotations
import os
from google import genai


class GeminiLLM:
    def __init__(self, model: str = "gemini-3-pro-preview"):
        # Uses GEMINI_API_KEY from environment
        self.client = genai.Client()
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # Deterministic-ish: temperature=0
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            # New SDK supports generation config; keep minimal for compatibility
            # If your SDK version supports config=..., you can add it.
        )
        return (resp.text or "").strip()
