import os
import json
from datetime import datetime

from src.data_loader import load_all_subsets_and_aggregate, SUBSETS
from src.graphrag_runner import build_graphrag, graphrag_retrieve_context, format_generation_prompt


def main():
    out_path = "outputs/graphrag_predictions.jsonl"
    os.makedirs("outputs", exist_ok=True)

    config = {
        "timestamp": datetime.now().isoformat(),
        "group": "graphrag",
        "implementation": "nano-graphrag",
        "llm_model": os.environ.get("GRAPHRAG_QWEN_MODEL", "qwen-plus"),
        "embedding_model": "BAAI/bge-m3",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k_equiv": 10,
        "retrieval_mode": "local",
        "context_budget_tokens_est": 5120,
    }
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
            rag.insert(docs)

            for i, (sample_id, q, gold) in enumerate(zip(agg.doc_ids, agg.questions, agg.answers_list)):
                ctx = graphrag_retrieve_context(rag, q, mode=config["retrieval_mode"])
                prompt = format_generation_prompt(q, ctx)

                from src.qwen_llm import QwenLLM
                llm = QwenLLM(model=os.environ.get("GRAPHRAG_QWEN_MODEL", "qwen-plus"))
                pred = llm.generate(prompt, temperature=0.0)

                rec = {
                    "subset": subset,
                    "sample_id": sample_id,
                    "sample_idx": i,
                    "question": q,
                    "answers": gold,
                    "pred_graphrag": pred,
                    "retrieved_context": ctx,
                    "config": config,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                print(f"  - [{subset}] sample {i+1}/10 done")

    print("\nGraphRAG run complete.")


if __name__ == "__main__":
    main()
