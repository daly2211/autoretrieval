"""Experiment — the script you edit.

Define your RAG pipeline here: how to chunk, embed, index, and retrieve.
The eval (run_eval.py) scores the results — it doesn't care how you got them.

You must export:
    get_retrieval_pipeline()  -> callable(corpora, questions_df, n) -> dict

    Returned dict must have:
        all_chunks:        {corpus_id: [chunk_text, ...]}
        retrieved_chunks:  {corpus_id: {question_idx: [chunk_text, ...]}}

python run_eval.py            # full eval
python run_eval.py --pct 10   # 10% sample for quick iteration
"""

import os
import re
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

from chunking_eval.utils import _safe_name

load_dotenv()

# Config

EMBEDDING_MODEL = "openai/text-embedding-3-large"

# Small model for keyword extraction — keeps costs minimal
KEYWORD_MODEL = "openai/gpt-4o-mini"

# Chunkers — define your splitting strategies here

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass


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


# Keyword extraction — uses a small LLM to pull search terms from questions

_KEYWORD_PROMPT = (
    "Extract 1-3 key search terms from this question. "
    "Return only the terms as a comma-separated list, no other text.\n\n"
    "Question: {question}"
)


def _extract_keywords(client: OpenAI, question: str) -> List[str]:
    response = client.chat.completions.create(
        model=KEYWORD_MODEL,
        messages=[{"role": "user", "content": _KEYWORD_PROMPT.format(question=question)}],
        max_tokens=30,
        temperature=0,
    )
    text = response.choices[0].message.content or ""
    return [t.strip().lower() for t in text.split(",") if t.strip()]


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
    llm_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def pipeline(
        corpora: Dict[str, str], questions_df: pd.DataFrame, n: int = 5,
    ) -> Dict[str, Any]:
        client = chromadb.EphemeralClient()

        all_chunks: Dict[str, List[str]] = {}
        retrieved_chunks: Dict[str, Dict[int, List[str]]] = {}

        for corpus_id, corpus_text in corpora.items():
            collection_name = _safe_name(corpus_id)
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass
            collection = client.create_collection(
                collection_name, embedding_function=ef,
                metadata={"hnsw:search_ef": 50})

            chunks_raw = chunker.split_text(corpus_text)
            all_chunks[corpus_id] = chunks_raw
            chunks_lower = [c.lower() for c in chunks_raw]

            batch_ids = [str(i) for i in range(len(chunks_lower))]
            batch_metas = [{"chunk": c} for c in chunks_raw]
            for j in range(0, len(chunks_lower), 500):
                collection.add(
                    documents=chunks_lower[j:j + 500],
                    metadatas=batch_metas[j:j + 500],
                    ids=batch_ids[j:j + 500],
                )

            corpus_questions = questions_df[questions_df['corpus_id'] == corpus_id]
            if corpus_questions.empty:
                retrieved_chunks[corpus_id] = {}
                continue

            retrieved_chunks[corpus_id] = {}
            for row_idx, row in corpus_questions.iterrows():
                question = row['question']
                keywords = _extract_keywords(llm_client, question)

                where_doc = None
                if len(keywords) == 1:
                    where_doc = {"$contains": keywords[0]}
                elif len(keywords) > 1:
                    where_doc = {"$or": [{"$contains": kw} for kw in keywords]}

                results = collection.query(
                    query_texts=[question],
                    n_results=n,
                    where_document=where_doc,
                )
                metas = results['metadatas'][0] if results['metadatas'] else []
                docs = [m['chunk'] for m in metas]
                retrieved_chunks[corpus_id][int(row_idx)] = docs

        return {
            'all_chunks': all_chunks,
            'retrieved_chunks': retrieved_chunks,
        }

    return pipeline
