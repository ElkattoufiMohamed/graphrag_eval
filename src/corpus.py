from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re

from src.data_loader import AggregatedCorpus


@dataclass(frozen=True)
class Chunk:
    subset: str
    doc_id: str
    chunk_id: str
    text: str


def _simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer (good enough for a baseline index builder).
    If you later switch to a true tokenizer (tiktoken / transformers), keep the interface.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ") if text else []


def _detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def chunk_document_tokens(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    Chunk a document by 'token' count (here: whitespace tokens).
    Produces overlapping chunks:
      start = 0 -> chunk_size
      start = chunk_size - overlap -> next window
    """
    assert chunk_size > 0
    assert 0 <= overlap < chunk_size

    tokens = _simple_tokenize(text)
    chunks: List[str] = []

    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(_detokenize(window))
        start += step

    return chunks


def build_unified_chunk_index(
    agg: AggregatedCorpus,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Builds the SINGLE unified chunk pool ("unified index") for one subset:
    - Input: 10 docs (agg.documents)
    - Output: list of Chunk objects across all docs
    """
    out: List[Chunk] = []

    for doc_idx, (doc_id, doc_text) in enumerate(zip(agg.doc_ids, agg.documents)):
        doc_chunks = chunk_document_tokens(doc_text, chunk_size=chunk_size, overlap=overlap)

        for j, ch in enumerate(doc_chunks):
            out.append(
                Chunk(
                    subset=agg.subset,
                    doc_id=doc_id,
                    chunk_id=f"{agg.subset}-doc{doc_idx}-chunk{j}",
                    text=ch,
                )
            )

    return out


def summarize_index(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Small helper to sanity-check the index.
    """
    by_doc: Dict[str, int] = {}
    total_tokens = 0

    for c in chunks:
        by_doc[c.doc_id] = by_doc.get(c.doc_id, 0) + 1
        total_tokens += len(_simple_tokenize(c.text))

    return {
        "num_chunks": len(chunks),
        "chunks_per_doc": by_doc,
        "approx_total_tokens": total_tokens,
        "avg_chunk_tokens": (total_tokens / max(len(chunks), 1)),
    }
