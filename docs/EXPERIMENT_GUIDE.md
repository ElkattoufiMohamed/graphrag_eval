# Experiment Guide (Step-by-step)

## 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure API keys
Copy the template and fill your keys:
```bash
cp .env.example .env
```

Then edit `.env` and set:
- `EVAL_LLM_PROVIDER=qwen` and `DASHSCOPE_API_KEY=...`, **or**
- `EVAL_LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`

If you choose OpenAI embeddings:
- `EVAL_EMBEDDING_BACKEND=openai`
- `OPENAI_API_KEY=...`

Otherwise keep:
- `EVAL_EMBEDDING_BACKEND=st` (local `BAAI/bge-m3`)

## 3) Run the full experiment
One command:
```bash
./scripts/run_experiment.sh
```

Or manually:
```bash
python -m src.run_baseline
python -m src.run_graphrag
python -m src.eval_compare
python -m src.build_report
```

## 4) Inspect outputs
- `outputs/baseline_predictions.jsonl`
- `outputs/graphrag_predictions.jsonl`
- `outputs/performance_comparison.csv`
- `outputs/final_report.md`

## 5) Notes
- The dataset subsets are: `musique`, `2wikimqa`, `narrativeqa`, `qasper`.
- Each subset uses top-10 samples aggregated into one unified pool.
- Fixed parameters: chunk size 512, overlap 50, top-k 10.
