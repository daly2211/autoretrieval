"""Generate a synthetic evaluation dataset from domain-specific documents via OpenRouter.

Just run: python generate_dataset.py
Edit the config section below to change corpus, output path, number of queries, etc.

Uses GPT-4-turbo to generate questions with ground-truth highlights, then filters for quality
and deduplicates. All API calls go through OpenRouter.

Requires OPENROUTER_API_KEY environment variable.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from chunking_eval import SyntheticEvaluation

load_dotenv()
# CONFIG — change these values to generate different datasets

# Path to corpus text file, or directory of .txt files
CORPUS = "./domain_specific_example/nvidia_10k.txt"

# Output CSV path for generated questions + references
OUTPUT_CSV = "./domain_specific_example/generated_queries_and_excerpts.csv"

# Number of queries to generate per corpus file
QUERIES_PER_CORPUS = 10

# Number of generation rounds (each round generates QUERIES_PER_CORPUS more)
ROUNDS = 1

# LLM model for question generation (OpenRouter format)
MODEL = "openai/gpt-4-turbo"

# Use approximate excerpt matching (chunk-based) instead of exact text search
APPROXIMATE_EXCERPTS = True

# Semantic similarity threshold — references below this are filtered out
FILTER_THRESHOLD = 0.36

# Duplicate threshold — questions more similar than this are removed
DUP_THRESHOLD = 0.7

# Set to True to skip the filtering steps (keep all generated questions)
SKIP_FILTER = False

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-large"


class OpenRouterSyntheticEvaluation(SyntheticEvaluation):
    def __init__(self, corpora_paths, queries_csv_path, chroma_db_path=None,
                 openrouter_api_key=None, openrouter_base_url=OPENROUTER_BASE_URL,
                 model="openai/gpt-4-turbo"):
        super().__init__(corpora_paths, queries_csv_path, chroma_db_path,
                         openai_api_key=openrouter_api_key, model=model)
        self.client = OpenAI(api_key=openrouter_api_key, base_url=openrouter_base_url)

    def _get_sim(self, target, references):
        response = self.client.embeddings.create(
            input=[target] + references,
            model=OPENROUTER_EMBEDDING_MODEL,
        )
        nparray1 = np.array(response.data[0].embedding)
        full_sim = []
        for i in range(1, len(response.data)):
            nparray2 = np.array(response.data[i].embedding)
            cosine_similarity = np.dot(nparray1, nparray2) / (
                np.linalg.norm(nparray1) * np.linalg.norm(nparray2))
            full_sim.append(cosine_similarity)
        return full_sim

    def _corpus_filter_duplicates(self, corpus_id, synth_questions_df, threshold):
        corpus_questions_df = synth_questions_df[synth_questions_df['corpus_id'] == corpus_id].copy()
        count_before = len(corpus_questions_df)
        corpus_questions_df.drop_duplicates(subset='question', keep='first', inplace=True)
        questions = corpus_questions_df['question'].tolist()

        response = self.client.embeddings.create(
            input=questions,
            model=OPENROUTER_EMBEDDING_MODEL,
        )
        embeddings_matrix = np.array([data.embedding for data in response.data])
        dot_product_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)

        def filter_vectors(sim_matrix, threshold):
            n = sim_matrix.shape[0]
            remaining = np.ones(n, dtype=bool)
            for i in range(n):
                if remaining[i]:
                    for j in range(i + 1, n):
                        if remaining[j] and sim_matrix[i, j] > threshold:
                            remaining[j] = 0
            return remaining

        rows_to_keep = filter_vectors(dot_product_matrix, threshold)
        corpus_questions_df = corpus_questions_df[rows_to_keep]
        count_after = len(corpus_questions_df)

        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")

        corpus_questions_df['references'] = corpus_questions_df['references'].apply(json.dumps)

        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[full_questions_df['corpus_id'] != corpus_id]
        full_questions_df = pd.concat([full_questions_df, corpus_questions_df], ignore_index=True)
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)
        full_questions_df.to_csv(self.questions_csv_path, index=False)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(CORPUS):
        import glob
        corpora_paths = sorted(glob.glob(os.path.join(CORPUS, "*.txt")))
        if not corpora_paths:
            print(f"Error: No .txt files found in {CORPUS}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(corpora_paths)} corpus files in {CORPUS}")
    else:
        corpora_paths = [CORPUS]

    print(f"Model: {MODEL}")
    print(f"Corpora: {corpora_paths}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Queries per corpus: {QUERIES_PER_CORPUS}")
    print(f"Rounds: {ROUNDS}")
    print(f"Approximate excerpts: {APPROXIMATE_EXCERPTS}")
    print()

    pipeline = OpenRouterSyntheticEvaluation(
        corpora_paths, OUTPUT_CSV,
        openrouter_api_key=api_key,
        openrouter_base_url=OPENROUTER_BASE_URL,
        model=MODEL,
    )

    print("=== Generating queries and excerpts ===")
    pipeline.generate_queries_and_excerpts(
        approximate_excerpts=APPROXIMATE_EXCERPTS,
        num_rounds=ROUNDS,
        queries_per_corpus=QUERIES_PER_CORPUS,
    )

    if not SKIP_FILTER:
        print("\n=== Filtering poor excerpts ===")
        pipeline.filter_poor_excerpts(threshold=FILTER_THRESHOLD)

        print("\n=== Removing duplicates ===")
        pipeline.filter_duplicates(threshold=DUP_THRESHOLD)

    df = pd.read_csv(OUTPUT_CSV)
    print(f"\nDone! {len(df)} questions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
