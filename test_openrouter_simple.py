from src.openrouter_llm import OpenRouterLLM

llm = OpenRouterLLM(model="qwen/qwen-plus")
print(llm.generate("Say 'ok' only.", temperature=0.0, max_tokens=10))
