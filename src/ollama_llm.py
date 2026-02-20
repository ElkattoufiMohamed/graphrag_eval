from __future__ import annotations

import os
import requests


class OllamaLLM:
    def __init__(self, model: str = "qcwind/qwen2.5-7B-instruct-Q4_K_M"):
        self.model = model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.timeout_s = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": float(temperature),
            },
        }

        resp = requests.post(url, json=payload, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        # Ollama returns eval/prompt counts in various fields depending on version.
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        message = data.get("message", {})
        return str(message.get("content", "")).strip()
