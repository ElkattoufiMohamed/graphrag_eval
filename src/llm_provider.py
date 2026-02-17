from __future__ import annotations

import os

from src.gemini_llm import GeminiLLM
from src.qwen_llm import QwenLLM


class UnifiedLLM:
    """Single configurable LLM provider for both baseline and GraphRAG runs."""

    def __init__(self, provider: str, model: str):
        provider_key = provider.strip().lower()
        if provider_key == "gemini":
            self.impl = GeminiLLM(model=model)
        elif provider_key == "qwen":
            self.impl = QwenLLM(model=model)
        else:
            raise ValueError("provider must be one of: gemini, qwen")

        self.provider = provider_key
        self.model = model
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        out = self.impl.generate(prompt, temperature=temperature)
        self.last_usage = getattr(
            self.impl,
            "last_usage",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        return out


def build_unified_llm_from_env() -> UnifiedLLM:
    provider = os.getenv("EVAL_LLM_PROVIDER", "qwen")

    default_model = {
        "gemini": "gemini-1.5-pro",
        "qwen": "qwen-plus",
    }.get(provider.strip().lower(), "qwen-plus")

    model = os.getenv("EVAL_LLM_MODEL", default_model)
    return UnifiedLLM(provider=provider, model=model)
