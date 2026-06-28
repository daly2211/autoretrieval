"""Experiment — the script you edit.

Define your RAG pipeline here: how to chunk, embed, index, and retrieve.
The eval (run_eval.py) scores the results — it doesn't care how you got them.

You must export:
    get_retrieval_pipeline()  -> callable(corpora, questions_df, n) -> dict

    Returned dict must have:
        all_chunk_ranges:   {corpus_id: [(start, end), ...]}
        retrieved_ranges:   {corpus_id: {question_idx: [(start, end), ...]}}

python run_eval.py            # full eval
python run_eval.py --pct 10   # 10% sample for quick iteration
"""

import os
import re
import sys
from typing import Any, Callable, Dict, List, Tuple

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from chunking_eval import BaseChunker
from chunking_eval.utils import rigorous_document_search

load_dotenv()

# Config

EMBEDDING_MODEL = "openai/text-embedding-3-large"

# Chunkers — define your splitting strategies here

class SentenceChunker(BaseChunker):
    """Split text on sentence boundaries, grouping N sentences per chunk."""

    def __init__(self, sentences_per_chunk: int = 10) -> None:
        self.sentences_per_chunk = sentences_per_chunk

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: List[str] = []
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + self.sentences_per_chunk])
            chunks.append(chunk)
        return chunks


# Embedding

def _get_embedding_function(
    model_name: str, api_key: str,
) -> embedding_functions.OpenAIEmbeddingFunction:
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model_name,
        api_base="https://openrouter.ai/api/v1",
    )

# Retrieval pipeline — the thing the eval scores

def get_retrieval_pipeline() -> Callable:
    """Build the default chromadb-based retrieval pipeline.

    Replace this entirely for custom retrieval strategies.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    chunker = SentenceChunker(sentences_per_chunk=10)
    ef = _get_embedding_function(EMBEDDING_MODEL, api_key)

    def pipeline(
        corpora: Dict[str, str], questions_df: pd.DataFrame, n: int = 5,
    ) -> Dict[str, Any]:
        client = chromadb.EphemeralClient()

        all_chunk_ranges: Dict[str, List[Tuple[int, int]]] = {}
        retrieved_ranges: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}

        for corpus_id, corpus_text in corpora.items():
            collection_name = _safe_name(corpus_id)
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            collection = client.create_collection(
                collection_name, embedding_function=ef,
                metadata={"hnsw:search_ef": 50})

            chunks = chunker.split_text(corpus_text)
            chunk_ranges: List[Tuple[int, int]] = []
            batch_docs: List[str] = []
            batch_metas: List[Dict[str, Any]] = []
            batch_ids: List[str] = []

            for i, chunk_text in enumerate(chunks):
                _, start, end = rigorous_document_search(corpus_text, chunk_text)
                chunk_ranges.append((start, end))
                batch_docs.append(chunk_text)
                batch_metas.append(
                    {"start_index": start, "end_index": end, "corpus_id": corpus_id})
                batch_ids.append(str(i))

            all_chunk_ranges[corpus_id] = chunk_ranges

            for j in range(0, len(batch_docs), 500):
                collection.add(
                    documents=batch_docs[j:j + 500],
                    metadatas=batch_metas[j:j + 500],
                    ids=batch_ids[j:j + 500],
                )

            corpus_questions = questions_df[questions_df['corpus_id'] == corpus_id]
            if corpus_questions.empty:
                retrieved_ranges[corpus_id] = {}
                continue

            results = collection.query(
                query_texts=corpus_questions['question'].tolist(),
                n_results=n,
            )

            metas_list = results['metadatas'] or []
            retrieved_ranges[corpus_id] = {}
            for row_idx, metas in zip(corpus_questions.index, metas_list):
                retrieved_ranges[corpus_id][int(row_idx)] = [
                    (int(m['start_index']), int(m['end_index']))
                    for m in metas
                ]

        return {
            'all_chunk_ranges': all_chunk_ranges,
            'retrieved_ranges': retrieved_ranges,
        }

    return pipeline


def _safe_name(path: str) -> str:
    name = path.replace(".", "_").replace("/", "_").replace("\\", "_").strip("_")[:60]
    if not name or name[0] not in "abcdefghijklmnopqrstuvwxyz0123456789":
        name = "c_" + name
    return name
