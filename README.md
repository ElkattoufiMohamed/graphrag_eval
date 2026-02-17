# GraphRAG Evaluation Pipeline (Internship Guide Aligned)

This repository evaluates **GraphRAG (nano-graphrag)** vs **Standard Vector RAG** on LongBench subsets:
- `musique`
- `2wikimqa` (WikiMQA in LongBench config naming)
- `narrativeqa`
- `qasper`

Each subset uses **top 10 test samples**, and the 10 contexts are aggregated into one unified retrieval pool per subset.

## Fixed Experiment Settings
- Chunk size: `512`
- Chunk overlap: `50`
- Retrieval top-k: `10`
- Metrics: `F1-Score` and `ROUGE-L`

## Unified Model Constraint
Use one LLM across both groups:
- `EVAL_LLM_PROVIDER=qwen` with `EVAL_LLM_MODEL=qwen-plus`, or
- `EVAL_LLM_PROVIDER=gemini` with `EVAL_LLM_MODEL=gemini-1.5-pro`

Embedding model (fixed for both groups):
- `EVAL_EMBEDDING_BACKEND=st` => `BAAI/bge-m3` (default), or
- `EVAL_EMBEDDING_BACKEND=openai` => `text-embedding-3-small`

## Run

```bash
python -m src.run_baseline
python -m src.run_graphrag
python -m src.eval_compare
python -m src.build_report
```

Outputs:
- `outputs/baseline_predictions.jsonl`
- `outputs/graphrag_predictions.jsonl`
- `outputs/performance_comparison.csv`
- `outputs/final_report.md`
