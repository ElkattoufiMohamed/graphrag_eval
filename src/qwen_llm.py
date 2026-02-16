import os
import dashscope
from dashscope import Generation

class QwenLLM:
    def __init__(self, model: str = "qwen-plus"):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY env var.")
        dashscope.api_key = api_key
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        resp = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            result_format="text",
        )
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"DashScope error: {resp}")
        return resp.output.text.strip()
