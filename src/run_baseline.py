from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any

from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.baseline_rag import make_embedder, build_baseline_index, retrieve_topk, build_prompt_from_chunks
from src.gemini_llm import GeminiLLM


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    # ----- Fixed experiment config (from doc) -----
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 10

    # Embeddings: choose ONE and keep it fixed for baseline + GraphRAG
    # Option 1 (local): BAAI/bge-m3
    embedder = make_embedder("st", model_name="BAAI/bge-m3")

    # LLM: choose ONE and keep it fixed for baseline + GraphRAG
    # Use a valid model code from Gemini docs/models page (example below).
    llm = GeminiLLM(model="gemini-3-flash-preview")

    # Load aggregated corpora (10 docs per subset)
    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    out_dir = "outputs"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "baseline_predictions.jsonl")

    run_meta: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "llm_model": llm.model,
        "embedding_model": "BAAI/bge-m3",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
    }

    print("Running baseline with config:", run_meta)
    print(f"Writing predictions to: {out_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        # For each subset: build index once, then run 10 queries
        for subset, agg in aggregated.items():
            print(f"\n[Baseline] Building index for subset={subset} ...")
            index = build_baseline_index(
                agg,
                embedder=embedder,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            for i, (qid, question, answers) in enumerate(zip(agg.doc_ids, agg.questions, agg.answers_list)):
                retrieved = retrieve_topk(index, embedder=embedder, query=question, top_k=TOP_K)
                prompt = build_prompt_from_chunks(question, retrieved)

                pred = llm.generate(prompt, temperature=0.0)

                record = {
                    "run_meta": run_meta,
                    "subset": subset,
                    "sample_id": qid,
                    "sample_idx": i,
                    "question": question,
                    "answers": answers,
                    "pred_baseline": pred,
                    "retrieved": [
                        {
                            "doc_id": r.chunk.doc_id,
                            "chunk_id": r.chunk.chunk_id,
                            "score": round(r.score, 6),
                            "text_preview": r.chunk.text[:200].replace("\n", " "),
                        }
                        for r in retrieved
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"  - [{subset}] sample {i+1}/10 done")

    print("\nBaseline run complete.")


if __name__ == "__main__":
    main()
