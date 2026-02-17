import os
import asyncio
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs

# -------------------------
# Chunking (token-based) to match baseline hyperparams: 512 / 50
# nano-graphrag passes token IDs, we return chunk records.
# -------------------------
def chunking_by_token_size(
    tokens_list: List[List[int]],
    doc_keys: List[str],
    tiktoken_model,
    overlap_token_size: int = 50,
    max_token_size: int = 512,
):
    results = []
    for doc_i, tokens in enumerate(tokens_list):
        chunk_token_batches = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk_ids = tokens[start : start + max_token_size]
            chunk_token_batches.append(chunk_ids)
            lengths.append(min(max_token_size, len(tokens) - start))

        chunk_texts = tiktoken_model.decode_batch(chunk_token_batches)
        for j, chunk in enumerate(chunk_texts):
            results.append(
                {
                    "tokens": lengths[j],
                    "content": chunk.strip(),
                    "chunk_order_index": j,
                    "full_doc_id": doc_keys[doc_i],
                }
            )
    return results


# -------------------------
# Embeddings: BAAI/bge-m3 (recommended)
# -------------------------
@dataclass
class LocalBGEEmbedder:
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    cache_dir: Optional[str] = None

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device, cache_folder=self.cache_dir)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.max_len = getattr(self.model, "max_seq_length", 8192)

    def encode(self, texts: List[str]) -> np.ndarray:
        # normalize_embeddings=True => cosine similarity friendly
        return self.model.encode(texts, normalize_embeddings=True)


# -------------------------
# LLM hook (async) - you already have src/gemini_llm.py
# nano-graphrag will call best_model_func / cheap_model_func many times during indexing.
# If your Gemini is unstable, you can later swap this to OpenAI or any provider.
# -------------------------
_LLM_SEM = asyncio.Semaphore(2)  # start small; raise if stable
_SYNC_LLM = None


def set_llm_for_graphrag(llm) -> None:
    global _SYNC_LLM
    _SYNC_LLM = llm

async def llm_complete(prompt: str, system_prompt=None, history_messages=None, **kwargs) -> str:
    if _SYNC_LLM is None:
        raise RuntimeError("GraphRAG LLM is not configured. Call set_llm_for_graphrag() first.")

    temperature = float(kwargs.get("temperature", 0.0))
    full_prompt = prompt if system_prompt is None else f"{system_prompt}\n\n{prompt}"

    async with _LLM_SEM:
        for attempt in range(8):
            try:
                def _call():
                    return _SYNC_LLM.generate(full_prompt, temperature=temperature)

                out = await asyncio.to_thread(_call)
                await asyncio.sleep(0.25 + random.uniform(0.0, 0.25))
                return out
            except Exception as e:
                msg = str(e)
                if ("429" in msg) or ("Too Many" in msg) or ("503" in msg) or ("temporarily" in msg.lower()):
                    wait = min(60, (2 ** attempt)) + random.uniform(0.0, 1.0)
                    await asyncio.sleep(wait)
                    continue
                raise


def build_graphrag(
    working_dir: str,
    embed_device: str = "cpu",
) -> GraphRAG:
    embedder = LocalBGEEmbedder(device=embed_device, cache_dir=working_dir)

    @wrap_embedding_func_with_attrs(
        embedding_dim=embedder.dim,
        max_token_size=embedder.max_len,
    )
    async def local_embedding(texts: List[str]) -> np.ndarray:
        return await asyncio.to_thread(embedder.encode, texts)

    rag = GraphRAG(
        working_dir=working_dir,
        # enforce baseline-style chunking
        chunk_func=chunking_by_token_size,
        # embeddings
        embedding_func=local_embedding,
        # LLMs: “best” and “cheap” (can be same for now)
        best_model_func=llm_complete,
        cheap_model_func=llm_complete,
    )
    return rag


def trim_context_to_budget(context: str, max_tokens_est: int = 5120) -> str:
    """
    Hard constraint in spec: GraphRAG retrieved context ~= baseline 10 chunks.
    We do a cheap estimate using whitespace tokens.
    """
    words = context.split()
    if len(words) <= max_tokens_est:
        return context
    return " ".join(words[:max_tokens_est])


def graphrag_retrieve_context(rag: GraphRAG, question: str, mode: str = "local") -> str:
    # only_need_context=True returns retrieved reports/entities instead of final answer
    qp = QueryParam(mode=mode, only_need_context=True)
    ctx = rag.query(question, param=qp)
    if not isinstance(ctx, str):
        ctx = str(ctx)
    return trim_context_to_budget(ctx, max_tokens_est=5120)


def format_generation_prompt(question: str, context: str) -> str:
    return (
        "You are answering using the provided context.\n"
        f"Question: {question}\n\n"
        "Context:\n\n"
        f"{context}\n\n"
        "Answer (be concise, factual, and only use the context):"
    )
