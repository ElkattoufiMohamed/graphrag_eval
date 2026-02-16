import os
import requests

class OpenRouterLLM:
    def __init__(self, model: str = "qwen/qwen-plus"):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY env var.")

        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # Optional attribution headers
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self.x_title = os.getenv("OPENROUTER_X_TITLE")

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")

        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
