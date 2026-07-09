# RAG Pipeline Benchmark

Benchmark chunking, embedding, and retrieval strategies against your own documents with ground-truth reference highlights.

## How it works

**`experiment.py`** — your pipeline: chunk, embed, index, retrieve. *Edit this.*
**`run_eval.py`** — scores whatever your pipeline returns. *Don't touch.*

```bash
python run_eval.py              # full eval
python run_eval.py --pct 10     # 10% sample (quick smoke test)
```

## Datasets

Two example datasets are included to show the pattern:

- **`domain_specific_example/`** — NVIDIA 10-K with auto-generated questions via `generate_dataset.py`
- **`general_evaluation_data/`** — 5 diverse documents (State of the Union, Wikipedia, chat logs, finance, biomedical) with pre-made questions

Neither is "the" dataset. Generate your own from any corpus:

```bash
python generate_dataset.py
```

Edit `CORPUS` and `OUTPUT_CSV` at the top of that script, then point `CORPUS_PATHS` and `QUESTIONS_CSV` in `run_eval.py` to match.

## Metrics

All scores are character-level overlap between retrieved chunks and ground-truth highlights.

| Metric | What it measures |
|---|---|
| **Recall** | How much of the relevant text was captured |
| **Precision** | How much of the retrieved text was actually relevant |
| **IoU** | Balanced single score — the best overall target |
| **Precision-Omega** | Chunker quality independent of retrieval |
