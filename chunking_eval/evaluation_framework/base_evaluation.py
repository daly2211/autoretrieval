import json
import os
import platform
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import chromadb
import numpy as np
import pandas as pd

from chunking_eval.utils import (
    rigorous_document_search, sum_of_ranges, union_ranges,
    intersect_two_ranges, difference, RangeTuple,
)


class BaseEvaluation:
    def __init__(self, questions_csv_path: str, chroma_db_path: Optional[str] = None,
                 corpora_id_paths: Optional[Dict[str, str]] = None) -> None:
        self.corpora_id_paths = corpora_id_paths
        self.questions_csv_path = questions_csv_path
        self.corpus_list: List[str] = []
        self._load_questions_df()

        if chroma_db_path is not None:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        else:
            self.chroma_client = chromadb.Client()

        self.is_general = False

    def _load_questions_df(self) -> None:
        if os.path.exists(self.questions_csv_path):
            self.questions_df = pd.read_csv(self.questions_csv_path)
            self.questions_df['references'] = self.questions_df['references'].apply(json.loads)
        else:
            self.questions_df = pd.DataFrame(columns=['question', 'references', 'corpus_id'])
        self.corpus_list = self.questions_df['corpus_id'].unique().tolist()

    def _get_chunks_and_metadata(self, splitter: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
        documents: List[str] = []
        metadatas: List[Dict] = []
        for corpus_id in self.corpus_list:
            corpus_path = corpus_id
            if self.corpora_id_paths is not None:
                corpus_path = self.corpora_id_paths[corpus_id]

            encoding: Optional[str] = 'utf-8' if platform.system() == 'Windows' else None
            with open(corpus_path, 'r', encoding=encoding) as file:
                corpus = file.read()

            current_documents = splitter.split_text(corpus)
            current_metadatas: List[Dict] = []
            for document in current_documents:
                try:
                    _, start_index, end_index = rigorous_document_search(corpus, document)
                except Exception:
                    raise Exception(f"Error finding '{document}' in {corpus_id}")
                current_metadatas.append({
                    "start_index": start_index,
                    "end_index": end_index,
                    "corpus_id": corpus_id,
                })
            documents.extend(current_documents)
            metadatas.extend(current_metadatas)
        return documents, metadatas

    def _full_precision_score(self, chunk_metadatas: List[Dict[str, Any]]) -> Tuple[List[float], List[int]]:
        ioc_scores: List[float] = []
        highlighted_chunks_count: List[int] = []
        for _, row in self.questions_df.iterrows():
            references = row['references']
            corpus_id = row['corpus_id']

            ioc_score = 0.0
            numerator_sets: List[RangeTuple] = []
            denominator_chunks_sets: List[RangeTuple] = []
            unused_highlights: List[RangeTuple] = [
                (x['start_index'], x['end_index']) for x in references
            ]
            highlighted_chunk_count = 0

            for metadata in chunk_metadatas:
                chunk_start, chunk_end, chunk_corpus_id = (
                    metadata['start_index'], metadata['end_index'], metadata['corpus_id'])
                if chunk_corpus_id != corpus_id:
                    continue

                contains_highlight = False
                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges(
                        (chunk_start, chunk_end), (ref_start, ref_end))
                    if intersection is not None:
                        contains_highlight = True
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges(
                            [(chunk_start, chunk_end)] + denominator_chunks_sets)

                if contains_highlight:
                    highlighted_chunk_count += 1

            highlighted_chunks_count.append(highlighted_chunk_count)
            denominator_sets = union_ranges(denominator_chunks_sets + unused_highlights)
            if numerator_sets:
                ioc_score = sum_of_ranges(numerator_sets) / sum_of_ranges(denominator_sets)
            ioc_scores.append(ioc_score)

        return ioc_scores, highlighted_chunks_count

    def _scores_from_dataset_and_retrievals(
        self, question_metadatas: List[List[Dict]], highlighted_chunks_count: List[int]
    ) -> Tuple[List[float], List[float], List[float]]:
        iou_scores: List[float] = []
        recall_scores: List[float] = []
        precision_scores: List[float] = []

        for (_, row), highlighted_chunk_count, metadatas in zip(
                self.questions_df.iterrows(), highlighted_chunks_count, question_metadatas):
            references = row['references']
            corpus_id = row['corpus_id']

            numerator_sets: List[RangeTuple] = []
            denominator_chunks_sets: List[RangeTuple] = []
            unused_highlights: List[RangeTuple] = [
                (x['start_index'], x['end_index']) for x in references
            ]

            for metadata in metadatas[:highlighted_chunk_count]:
                chunk_start, chunk_end, chunk_corpus_id = (
                    metadata['start_index'], metadata['end_index'], metadata['corpus_id'])
                if chunk_corpus_id != corpus_id:
                    continue

                for ref_obj in references:
                    ref_start, ref_end = int(ref_obj['start_index']), int(ref_obj['end_index'])
                    intersection = intersect_two_ranges(
                        (chunk_start, chunk_end), (ref_start, ref_end))
                    if intersection is not None:
                        unused_highlights = difference(unused_highlights, intersection)
                        numerator_sets = union_ranges([intersection] + numerator_sets)
                        denominator_chunks_sets = union_ranges(
                            [(chunk_start, chunk_end)] + denominator_chunks_sets)

            numerator_value = sum_of_ranges(numerator_sets) if numerator_sets else 0
            recall_denominator = sum_of_ranges(
                [(x['start_index'], x['end_index']) for x in references])
            precision_denominator = sum_of_ranges(
                [(x['start_index'], x['end_index']) for x in metadatas[:highlighted_chunk_count]])
            iou_denominator = precision_denominator + sum_of_ranges(unused_highlights)

            recall_scores.append(numerator_value / recall_denominator)
            precision_scores.append(numerator_value / precision_denominator)
            iou_scores.append(numerator_value / iou_denominator)

        return iou_scores, recall_scores, precision_scores

    def _chunker_to_collection(
        self, chunker: Any, embedding_function: Any,
        chroma_db_path: Optional[str] = None, collection_name: Optional[str] = None,
    ) -> chromadb.Collection:
        collection: Optional[chromadb.Collection] = None
        if chroma_db_path is not None:
            try:
                chunk_client = chromadb.PersistentClient(path=chroma_db_path)
                if collection_name is not None:
                    collection = chunk_client.create_collection(
                        collection_name, embedding_function=embedding_function,
                        metadata={"hnsw:search_ef": 50})
            except Exception:
                pass

        collection_name = collection_name or "auto_chunk"
        if collection is None:
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass
            collection = self.chroma_client.create_collection(
                collection_name, embedding_function=embedding_function,
                metadata={"hnsw:search_ef": 50})

        docs, metas = self._get_chunks_and_metadata(chunker)
        BATCH_SIZE = 500
        for i in range(0, len(docs), BATCH_SIZE):
            batch_docs = docs[i:i + BATCH_SIZE]
            batch_metas = metas[i:i + BATCH_SIZE]
            batch_ids = [str(j) for j in range(i, i + len(batch_docs))]
            collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
        return collection

    def run(
        self, chunker: Any, embedding_function: Any = None,
        retrieve: int = 5, db_to_save_chunks: Optional[str] = None,
    ) -> Dict[str, Union[float, Dict]]:
        self._load_questions_df()
        if embedding_function is None:
            raise ValueError("embedding_function is required")

        collection: Optional[chromadb.Collection] = None
        if db_to_save_chunks is not None:
            chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "0"
            chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "0"
            embedding_function_name = embedding_function.__class__.__name__
            if embedding_function_name == "SentenceTransformerEmbeddingFunction":
                embedding_function_name = "SentEmbFunc"
            collection_name = (
                f"{embedding_function_name}_{chunker.__class__.__name__}_"
                f"{int(float(chunk_size))}_{int(float(chunk_overlap))}")
            try:
                chunk_client = chromadb.PersistentClient(path=db_to_save_chunks)
                collection = chunk_client.get_collection(
                    collection_name, embedding_function=embedding_function)
            except Exception:
                collection = self._chunker_to_collection(
                    chunker, embedding_function, chroma_db_path=db_to_save_chunks,
                    collection_name=collection_name)

        if collection is None:
            collection = self._chunker_to_collection(chunker, embedding_function)

        question_collection: Optional[chromadb.Collection] = None

        if self.is_general:
            import glob as _glob
            questions_db_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                                            'general_evaluation_data', 'questions_db')
            questions_db_dir = os.path.abspath(questions_db_dir)
            if os.path.exists(questions_db_dir):
                questions_client = chromadb.PersistentClient(path=questions_db_dir)
                ef_name = embedding_function.__class__.__name__
                try:
                    if ef_name == "OpenAIEmbeddingFunction":
                        if hasattr(embedding_function, 'model_name'):
                            mn = embedding_function.model_name
                        else:
                            mn = embedding_function._model_name
                        if "text-embedding-3-large" in mn:
                            question_collection = questions_client.get_collection(
                                "auto_questions_openai_large",
                                embedding_function=embedding_function)
                        elif "text-embedding-3-small" in mn:
                            question_collection = questions_client.get_collection(
                                "auto_questions_openai_small",
                                embedding_function=embedding_function)
                    elif ef_name == "SentenceTransformerEmbeddingFunction":
                        question_collection = questions_client.get_collection(
                            "auto_questions_sentence_transformer",
                            embedding_function=embedding_function)
                except Exception:
                    pass

        if not self.is_general or question_collection is None:
            try:
                self.chroma_client.delete_collection("auto_questions")
            except Exception:
                pass
            question_collection = self.chroma_client.create_collection(
                "auto_questions", embedding_function=embedding_function,
                metadata={"hnsw:search_ef": 50})
            question_collection.add(
                documents=self.questions_df['question'].tolist(),
                metadatas=[{"corpus_id": x} for x in self.questions_df['corpus_id'].tolist()],
                ids=[str(i) for i in self.questions_df.index],
            )

        question_db = question_collection.get(include=['embeddings'])
        question_ids: List[int] = [int(id) for id in question_db['ids']]
        sorted_pairs = sorted(zip(question_ids, question_db['embeddings'] or []))
        sorted_embeddings: List[np.ndarray] = [emb for _, emb in sorted_pairs]
        self.questions_df = self.questions_df.sort_index()

        brute_iou_scores, highlighted_chunks_count = self._full_precision_score(
            collection.get()['metadatas'] or [])

        if retrieve == -1:
            maximum_n = min(20, max(highlighted_chunks_count))
        else:
            highlighted_chunks_count = [retrieve] * len(highlighted_chunks_count)
            maximum_n = retrieve

        retrievals = collection.query(
            query_embeddings=list(sorted_embeddings), n_results=maximum_n)

        iou_scores, recall_scores, precision_scores = self._scores_from_dataset_and_retrievals(
            cast(List[List[Dict[str, Any]]], retrievals['metadatas'] or []),
            highlighted_chunks_count,
        )

        corpora_scores: Dict[str, Dict[str, List[float]]] = {}
        for index, row in self.questions_df.iterrows():
            corpus_id = row['corpus_id']
            if corpus_id not in corpora_scores:
                corpora_scores[corpus_id] = {
                    "precision_omega_scores": [], "iou_scores": [],
                    "recall_scores": [], "precision_scores": [],
                }
            corpora_scores[corpus_id]['precision_omega_scores'].append(brute_iou_scores[index])
            corpora_scores[corpus_id]['iou_scores'].append(iou_scores[index])
            corpora_scores[corpus_id]['recall_scores'].append(recall_scores[index])
            corpora_scores[corpus_id]['precision_scores'].append(precision_scores[index])

        return {
            "corpora_scores": corpora_scores,
            "iou_mean": np.mean(iou_scores),
            "iou_std": np.std(iou_scores),
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
            "precision_omega_mean": np.mean(brute_iou_scores),
            "precision_omega_std": np.std(brute_iou_scores),
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
        }
