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
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        resp = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            result_format="text",
        )
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(f"DashScope error: {resp}")

        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_toks = int(getattr(usage, "input_tokens", 0) or 0)
            completion_toks = int(getattr(usage, "output_tokens", 0) or 0)
            self.last_usage = {
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
            }
        return resp.output.text.strip()
