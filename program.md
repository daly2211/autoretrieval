# autoretrieval

## Goal

Improve the retrieval pipeline by editing `experiment.py`. The eval scores four metrics — **IoU is the best single target** since it balances recall and precision. Precision-Omega tells you how good your chunker is independent of retrieval.

The eval is configured to use:
- Corpus: `./domain_specific_example/nvidia_10k.txt`
- Questions: `./domain_specific_example/generated_queries_and_excerpts.csv`

**User constraints (fill in what matters for this run):**

<!-- User: add any constraints here. Examples:
     - Don't use OpenAI embeddings 
     - Don't change the keyword model
-->

## Files that concern you

| File | Role |
|---|---|
| `experiment.py` | **The file you edit.** Chunker, embedding, retrieval. |
| `run_eval.py` | Read-only scoring engine. Do not modify. |


## Pipeline contract

`experiment.py` exports `get_retrieval_pipeline()` which returns a callable that receives `corpora` (dict of path → text) and `questions_df` (pandas DataFrame). It must return:

```python
{
    "all_chunks":       {corpus_id: ["chunk text", ...]},
    "retrieved_chunks": {corpus_id: {question_idx: ["chunk text", ...]}},
}
```

## What you can change

Everything in `experiment.py`. In-code comments show alternate approaches:

- **Chunker** — any `BaseChunker` subclass. Sentence-based, character-based, recursive split, semantic clustering, LLM-based. A `SentenceChunker` and commented-out `CharChunker` are provided as starting points.
- **Chunk size** — `sentences_per_chunk`, `chunk_size`, etc. Smaller = precision. Larger = recall.
- **Embedding model** — `EMBEDDING_MODEL`. Any model Chroma supports: `text-embedding-3-large`, `text-embedding-3-small`, `all-MiniLM-L6-v2`, etc.
- **No embeddings** — drop `embedding_function`, use Chroma's `collection.get` with `where_document` for pure keyword search, or use an entirely different implementation.
- **Keyword filtering** — `USE_KEYWORD_FILTER` toggle. A small LLM extracts keywords per question and `where_document={"$contains": ...}` post-filters vector results.
- **Keyword model** — `KEYWORD_MODEL`. Swap to any OpenRouter model.
- **Retrieve count** — `n_results`. More chunks = higher recall but lower precision/IoU.
- **Vector store** — `EphemeralClient` vs `PersistentClient` (caches embeddings). Replace Chroma entirely — the eval only sees text.

## What you cannot change

- `run_eval.py` — scoring is fixed.
- `chunking_eval/` — vendored utils.
- The return format — `{all_chunks, retrieved_chunks}` with string values.

## Scoring

Character-level overlap between retrieved chunks and ground-truth highlights:

| Metric | What it measures |
|---|---|
| **IoU** | Overlap / union — balanced single score. Main optimization target. |
| **Recall** | Overlap / highlight length — did we miss relevant content? |
| **Precision** | Overlap / retrieved length — did we grab noise? |
| **Precision-Omega** | Precision ceiling for this chunker — diagnose chunker vs retrieval issues. |


## Output format

After each run, the eval prints:

```
Recall: 0.4800 +/- 0.4193
Precision: 0.0194 +/- 0.0182
Precision-Omega: 0.1325 +/- 0.0838
IoU: 0.0193 +/- 0.0183
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT commas).

The TSV has these columns:

```
commit	iou	precision_omega	recall	precision	status	description
```

1. git commit hash (short, 7 chars)
2. IoU (mean) — use 0.0000 for crashes
3. Precision-Omega (mean) — use 0.0000 for crashes
4. Recall (mean) — use 0.0000 for crashes
5. Precision (mean) — use 0.0000 for crashes
6. status: `keep`, `discard`, or `crash`
7. short description of what this experiment tried

Example:

```
commit	iou	precision_omega	recall	precision	status	description
a1b2c3d	0.0193	0.1325	0.4800	0.0194	keep	baseline (10 sentences, text-embedding-3-large, keyword filter)
b2c3d4e	0.0221	0.1412	0.5120	0.0201	keep	switch to 5 sentences per chunk
c3d4e5f	0.0180	0.1301	0.4600	0.0185	discard	switch to text-embedding-3-small
d4e5f6g	0.0000	0.0000	0.0000	0.0000	crash	remove keyword model entirely
```

Do not commit `results.tsv` — leave it untracked.

## Experiment loop

Run on a dedicated branch (e.g. `autoretrieval/jul8`).

LOOP FOREVER:

1. Look at git state: current branch/commit.
2. Change one thing in `experiment.py`. Chunk size, embedding model, keyword toggle, chunker type. One variable.
3. `git commit`
4. Run: `python run_eval.py`
5. Record the results in `results.tsv`.
6. If IoU improved → keep the commit. If equal or worse → `git reset --hard HEAD~1` to discard.
7. Repeat.

**First run**: run as-is to establish the baseline.

**Crashes**: if a run crashes, fix trivial bugs and re-run. If the idea itself is fundamentally broken, log "crash" and move on.

**NEVER STOP**: once the loop begins, do not pause to ask if you should continue. Run until interrupted. The human may be asleep. If you run out of ideas, re-read the files for new angles, try combining previous near-misses, try more radical changes.

**Simplicity**: all else equal, simpler is better. A tiny gain from hacky code is not worth it. A gain from deleting code is ideal.
