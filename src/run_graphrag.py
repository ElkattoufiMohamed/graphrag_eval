import os
import json
from datetime import datetime

from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.graphrag_runner import (
    build_graphrag,
    graphrag_retrieve_context,
    format_generation_prompt,
    set_llm_for_graphrag,
)
from src.llm_provider import build_unified_llm_from_env


def _approx_tokens(text: str) -> int:
    return len(text.split())


def main():
    out_path = "outputs/graphrag_predictions.jsonl"
    os.makedirs("outputs", exist_ok=True)

    config = {
        "timestamp": datetime.now().isoformat(),
        "group": "graphrag",
        "implementation": "nano-graphrag",
        "llm_provider": None,
        "llm_model": None,
        "embedding_model": None,
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k_equiv": 10,
        "retrieval_mode": "local",
        "context_budget_tokens_est": 5120,
    }

    llm = build_unified_llm_from_env()
    set_llm_for_graphrag(llm)
    config["llm_provider"] = llm.provider
    config["llm_model"] = llm.model

    embedding_backend = os.environ.get("EVAL_EMBEDDING_BACKEND", "st")
    config["embedding_model"] = "text-embedding-3-small" if embedding_backend == "openai" else "BAAI/bge-m3"
    print("Running GraphRAG with config:", config)
    print("Writing predictions to:", out_path)

    aggregated = load_all_subsets_and_aggregate(subsets=SUBSETS, k=10, split="test")

    with open(out_path, "w", encoding="utf-8") as f:
        for subset in SUBSETS:
            agg = aggregated[subset]
            docs = agg.documents  # 10 contexts (the mixed pool)

            workdir = f"outputs/nano_graphrag_cache/{subset}"
            os.makedirs(workdir, exist_ok=True)

            rag = build_graphrag(
                working_dir=workdir,
                embed_device=os.environ.get("EMBED_DEVICE", "cpu"),
            )

            print(f"\n[GraphRAG] Indexing subset={subset} with {len(docs)} docs ...")
            graph_construction_tokens_est = sum(_approx_tokens(doc) for doc in docs)
            rag.insert(docs)

            for i, (sample_id, q, gold) in enumerate(zip(agg.doc_ids, agg.questions, agg.answers_list)):
                ctx = graphrag_retrieve_context(rag, q, mode=config["retrieval_mode"])
                prompt = format_generation_prompt(q, ctx)

                pred = llm.generate(prompt, temperature=0.0)

                rec = {
                    "subset": subset,
                    "sample_id": sample_id,
                    "sample_idx": i,
                    "question": q,
                    "answers": gold,
                    "pred_graphrag": pred,
                    "retrieved_context": ctx,
                    "graph_construction_tokens_est": graph_construction_tokens_est,
                    "config": config,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                print(f"  - [{subset}] sample {i+1}/10 done")

    print("\nGraphRAG run complete.")


if __name__ == "__main__":
    main()
