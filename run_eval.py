"""Evaluation engine — does not change.

Scores a retrieval pipeline against a question+reference dataset.

To run:
    python run_eval.py              # full eval
    python run_eval.py --pct 10     # 10% sample for quick iteration

Your experiment module must export:
    get_retrieval_pipeline()  -> callable(corpora, questions, n) -> dict
    Returns {all_chunks, retrieved_chunks} where values are chunk text strings.
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from chunking_eval.utils import (
    RangeTuple, sum_of_ranges, union_ranges,
    intersect_two_ranges, difference, chunks_to_ranges,
)

load_dotenv()

# Eval settings — what data to score against

CORPUS_PATHS: List[str] = [
    "./domain_specific_example/nvidia_10k.txt",
]

QUESTIONS_CSV: str = "./domain_specific_example/generated_queries_and_excerpts.csv"

PCT: float = 100.0

# Token-level scoring

def _score_retrieval(
    retrieved_ranges: List[RangeTuple], reference_ranges: List[RangeTuple],
) -> Tuple[float, float, float]:
    numerator: List[RangeTuple] = []
    unused_refs: List[RangeTuple] = list(reference_ranges)
    raw_retrieved_len = sum_of_ranges(retrieved_ranges)

    for chunk_range in retrieved_ranges:
        for ref_range in reference_ranges:
            overlap = intersect_two_ranges(chunk_range, ref_range)
            if overlap is not None:
                unused_refs = difference(unused_refs, overlap)
                numerator = union_ranges([overlap] + numerator)

    num_len = sum_of_ranges(numerator)
    ref_len = sum_of_ranges(reference_ranges)

    recall = num_len / ref_len if ref_len > 0 else 0.0
    precision = num_len / raw_retrieved_len if raw_retrieved_len > 0 else 0.0
    iou_denom = raw_retrieved_len + sum_of_ranges(unused_refs)
    iou = num_len / iou_denom if iou_denom > 0 else 0.0

    return recall, precision, iou


def _compute_all_metrics(
    all_chunk_ranges: Dict[str, List[RangeTuple]],
    questions_df: pd.DataFrame,
    retrieval_fn: Callable[[int, str], List[RangeTuple]],
) -> Dict[str, float]:
    recall_scores: List[float] = []
    precision_scores: List[float] = []
    iou_scores: List[float] = []
    prec_omega_scores: List[float] = []

    for idx, row in questions_df.iterrows():
        references: list[Any] = row['references']
        corpus_id: str = row['corpus_id']

        ref_ranges: List[RangeTuple] = [(r['start_index'], r['end_index']) for r in references]
        retrieved = retrieval_fn(idx, corpus_id)

        recall, precision, iou = _score_retrieval(retrieved, ref_ranges)
        recall_scores.append(recall)
        precision_scores.append(precision)
        iou_scores.append(iou)

        omega_numerator: List[RangeTuple] = []
        omega_denominator: List[RangeTuple] = []
        unused_refs: List[RangeTuple] = list(ref_ranges)

        for chunk_range in all_chunk_ranges.get(corpus_id, []):
            for ref_range in ref_ranges:
                overlap = intersect_two_ranges(chunk_range, ref_range)
                if overlap is not None:
                    unused_refs = difference(unused_refs, overlap)
                    omega_numerator = union_ranges([overlap] + omega_numerator)
                    omega_denominator = union_ranges([chunk_range] + omega_denominator)

        num_len = sum_of_ranges(omega_numerator)
        omega_denom_len = sum_of_ranges(omega_denominator + unused_refs)
        prec_omega = num_len / omega_denom_len if omega_denom_len > 0 else 0.0
        prec_omega_scores.append(prec_omega)

    return {
        'recall_mean': float(np.mean(recall_scores)),
        'recall_std': float(np.std(recall_scores)),
        'precision_mean': float(np.mean(precision_scores)),
        'precision_std': float(np.std(precision_scores)),
        'iou_mean': float(np.mean(iou_scores)),
        'iou_std': float(np.std(iou_scores)),
        'precision_omega_mean': float(np.mean(prec_omega_scores)),
        'precision_omega_std': float(np.std(prec_omega_scores)),
    }


def _sample_questions(csv_path: str, pct: float, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['references'] = df['references'].apply(json.loads)
    total = len(df)
    if pct >= 100:
        print(f"Using all {total} questions")
        return df

    n = max(1, int(total * pct / 100))
    sampled = df.sample(n=n, random_state=seed)
    print(f"Sampled {n}/{total} questions ({pct}%)")
    return sampled


def _load_corpora(corpus_paths: List[str]) -> Dict[str, str]:
    corpora: Dict[str, str] = {}
    for path in corpus_paths:
        with open(path, 'r', encoding='utf-8') as f:
            corpora[path] = f.read()
    return corpora


def _print_metrics(results: Dict[str, float]) -> None:
    for metric in ['Recall', 'Precision', 'Precision-Omega', 'IoU']:
        key = metric.lower().replace('-', '_')
        print(f"{metric}: {results[f'{key}_mean']:.4f} +/- {results[f'{key}_std']:.4f}")


# Public entry point

def run_evaluation(
    retrieval_pipeline: Callable,
    *,
    corpus_paths: Optional[List[str]] = None,
    questions_csv: Optional[str] = None,
    pct: float = 100.0,
) -> Dict[str, float]:
    corpus_paths = corpus_paths or CORPUS_PATHS
    questions_csv = questions_csv or QUESTIONS_CSV

    if not os.path.exists(questions_csv):
        print(f"Error: {questions_csv} not found.", file=sys.stderr)
        sys.exit(1)

    questions_df = _sample_questions(questions_csv, pct)
    corpora = _load_corpora(corpus_paths)

    missing_corpora = set(questions_df['corpus_id']) - set(corpora)
    if missing_corpora:
        print(
            f"Error: corpus_id values in {questions_csv} not found in loaded "
            f"corpora {list(corpora.keys())}: {sorted(missing_corpora)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Questions: {len(questions_df)}")
    print(f"Corpora: {list(corpora.keys())}")
    print()

    pipeline_data = retrieval_pipeline(corpora, questions_df)

    all_chunk_ranges: Dict[str, List[RangeTuple]] = {}
    for corpus_id, chunks in pipeline_data['all_chunks'].items():
        all_chunk_ranges[corpus_id] = chunks_to_ranges(corpora[corpus_id], chunks)

    retrieved_ranges: Dict[str, Dict[int, List[RangeTuple]]] = {}
    for corpus_id, qmap in pipeline_data['retrieved_chunks'].items():
        retrieved_ranges[corpus_id] = {}
        for q_idx, chunks in qmap.items():
            retrieved_ranges[corpus_id][int(q_idx)] = chunks_to_ranges(
                corpora[corpus_id], chunks)

    def get_retrieved(idx: int, corpus_id: str) -> List[RangeTuple]:
        return retrieved_ranges.get(corpus_id, {}).get(int(idx), [])

    results = _compute_all_metrics(all_chunk_ranges, questions_df, get_retrieved)
    _print_metrics(results)
    return results


# CLI entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG chunking evaluation")
    parser.add_argument("--pct", type=float, default=PCT,
                        help=f"Percentage of eval questions to use (default: {PCT})")
    args = parser.parse_args()

    from experiment import get_retrieval_pipeline

    pipeline = get_retrieval_pipeline()
    run_evaluation(pipeline, pct=args.pct)


if __name__ == "__main__":
    main()
