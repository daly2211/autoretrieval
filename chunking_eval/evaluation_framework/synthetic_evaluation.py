import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from .base_evaluation import BaseEvaluation
from chunking_eval.utils import rigorous_document_search

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')
_PROMPTS_DIR = os.path.abspath(_PROMPTS_DIR)


def _read_prompt(filename: str) -> str:
    with open(os.path.join(_PROMPTS_DIR, filename), 'r') as f:
        return f.read()


class SyntheticEvaluation(BaseEvaluation):
    def __init__(self, corpora_paths: List[str], queries_csv_path: str,
                 chroma_db_path: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 model: str = "gpt-4-turbo") -> None:
        super().__init__(questions_csv_path=queries_csv_path, chroma_db_path=chroma_db_path)
        self.corpora_paths = corpora_paths
        self.questions_csv_path = queries_csv_path
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.synth_questions_df: Optional[pd.DataFrame] = None

        self.question_maker_system_prompt = _read_prompt('question_maker_system.txt')
        self.question_maker_approx_system_prompt = _read_prompt('question_maker_approx_system.txt')
        self.question_maker_user_prompt = _read_prompt('question_maker_user.txt')
        self.question_maker_approx_user_prompt = _read_prompt('question_maker_approx_user.txt')

    def _save_questions_df(self) -> None:
        if self.synth_questions_df is not None:
            self.synth_questions_df.to_csv(self.questions_csv_path, index=False)

    def _tag_text(self, text: str) -> Tuple[str, List[int]]:
        chunk_length = 100
        chunks: List[str] = []
        tag_indexes: List[int] = [0]
        start = 0
        while start < len(text):
            end = start + chunk_length
            chunk = text[start:end]
            if end < len(text):
                space_index = chunk.rfind(' ')
                if space_index != -1:
                    end = start + space_index + 1
                    chunk = text[start:end]
            chunks.append(chunk)
            tag_indexes.append(end)
            start = end

        tagged_text = ""
        for i, chunk in enumerate(chunks):
            tagged_text += f"<start_chunk_{i}>" + chunk + f"<end_chunk_{i}>"
        return tagged_text, tag_indexes

    def _extract_question_and_approx_references(
        self, corpus: str, document_length: int = 4000,
        prev_questions: Optional[List[str]] = None,
    ) -> Tuple[str, List[Tuple[str, int, int]]]:
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index:start_index + document_length]
        else:
            start_index = 0
            document = corpus

        if prev_questions:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        tagged_text, tag_indexes = self._tag_text(document)

        completion = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            max_tokens=600,
            messages=[
                {"role": "system", "content": self.question_maker_approx_system_prompt},
                {"role": "user", "content": self.question_maker_approx_user_prompt
                    .replace("{document}", tagged_text)
                    .replace("{prev_questions_str}", prev_questions_str)},
            ],
        )

        content = completion.choices[0].message.content or ""
        json_response = json.loads(content)

        if 'references' not in json_response:
            raise ValueError("The response does not contain a 'references' field.")
        if 'question' not in json_response:
            raise ValueError("The response does not contain a 'question' field.")

        text_references = json_response['references']
        question = json_response['question']

        references: List[Tuple[str, int, int]] = []
        for reference in text_references:
            reference_keys = list(reference.keys())
            if len(reference_keys) != 3:
                raise ValueError(
                    f"Each reference must have exactly 3 keys. Got keys: {reference_keys}")

            if 'end_chunk' not in reference_keys:
                reference_keys.remove('content')
                reference_keys.remove('start_chunk')
                end_chunk_key = reference_keys[0]
                end_index = start_index + tag_indexes[reference[end_chunk_key] + 1]
            else:
                end_index = start_index + tag_indexes[reference['end_chunk'] + 1]

            ref_start = start_index + tag_indexes[reference['start_chunk']]
            references.append((corpus[ref_start:end_index], ref_start, end_index))

        return question, references

    def _extract_question_and_references(
        self, corpus: str, document_length: int = 4000,
        prev_questions: Optional[List[str]] = None,
    ) -> Tuple[str, List[Tuple[str, int, int]]]:
        if len(corpus) > document_length:
            start_index = random.randint(0, len(corpus) - document_length)
            document = corpus[start_index:start_index + document_length]
        else:
            document = corpus

        if prev_questions:
            if len(prev_questions) > 20:
                questions_sample = random.sample(prev_questions, 20)
                prev_questions_str = '\n'.join(questions_sample)
            else:
                prev_questions_str = '\n'.join(prev_questions)
        else:
            prev_questions_str = ""

        completion = self.client.chat.completions.create(
            model="gpt-4-turbo",
            response_format={"type": "json_object"},
            max_tokens=600,
            messages=[
                {"role": "system", "content": self.question_maker_system_prompt},
                {"role": "user", "content": self.question_maker_user_prompt
                    .replace("{document}", document)
                    .replace("{prev_questions_str}", prev_questions_str)},
            ],
        )

        content = completion.choices[0].message.content or ""
        json_response = json.loads(content)

        if 'references' not in json_response:
            raise ValueError("The response does not contain a 'references' field.")
        if 'question' not in json_response:
            raise ValueError("The response does not contain a 'question' field.")

        text_references = json_response['references']
        question = json_response['question']

        references: List[Tuple[str, int, int]] = []
        for reference in text_references:
            if not isinstance(reference, str):
                raise ValueError(
                    f"Expected reference to be str, got {type(reference).__name__}")
            try:
                ref_text, ref_start, ref_end = rigorous_document_search(corpus, reference)
            except ValueError as e:
                raise ValueError(
                    f"No match found for reference.\nReference: {reference}") from e
            references.append((ref_text, ref_start, ref_end))

        return question, references

    def _generate_corpus_questions(
        self, corpus_id: str, approx: bool = False, n: int = 5,
    ) -> None:
        with open(corpus_id, 'r') as file:
            corpus = file.read()

        i = 0
        while i < n:
            while True:
                try:
                    print(f"Trying Query {i}")
                    questions_list: List[str] = []
                    if self.synth_questions_df is not None:
                        questions_list = self.synth_questions_df[
                            self.synth_questions_df['corpus_id'] == corpus_id
                        ]['question'].tolist()
                    if approx:
                        question, references = self._extract_question_and_approx_references(
                            corpus, 4000, questions_list)
                    else:
                        question, references = self._extract_question_and_references(
                            corpus, 4000, questions_list)

                    if len(references) > 5:
                        raise ValueError("The number of references exceeds 5.")

                    references_json: List[Dict[str, Any]] = [
                        {'content': ref[0], 'start_index': ref[1], 'end_index': ref[2]}
                        for ref in references
                    ]
                    new_question = {
                        'question': question,
                        'references': json.dumps(references_json),
                        'corpus_id': corpus_id,
                    }
                    new_df = pd.DataFrame([new_question])
                    if self.synth_questions_df is not None:
                        self.synth_questions_df = pd.concat(
                            [self.synth_questions_df, new_df], ignore_index=True)
                    else:
                        self.synth_questions_df = new_df
                    self._save_questions_df()
                    break
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Error occurred: {e}")
                    continue
            i += 1

    def _get_synth_questions_df(self) -> pd.DataFrame:
        if os.path.exists(self.questions_csv_path):
            return pd.read_csv(self.questions_csv_path)
        return pd.DataFrame(columns=['question', 'references', 'corpus_id'])

    def generate_queries_and_excerpts(
        self, approximate_excerpts: bool = False, num_rounds: int = -1,
        queries_per_corpus: int = 5,
    ) -> None:
        self.synth_questions_df = self._get_synth_questions_df()
        rounds = 0
        while num_rounds == -1 or rounds < num_rounds:
            for corpus_id in self.corpora_paths:
                self._generate_corpus_questions(
                    corpus_id, approx=approximate_excerpts, n=queries_per_corpus)
            rounds += 1

    def _get_sim(self, target: str, references: List[str]) -> List[float]:
        response = self.client.embeddings.create(
            input=[target] + references,
            model="text-embedding-3-large",
        )
        nparray1 = np.array(response.data[0].embedding)
        full_sim: List[float] = []
        for i in range(1, len(response.data)):
            nparray2 = np.array(response.data[i].embedding)
            cosine_similarity = np.dot(nparray1, nparray2) / (
                np.linalg.norm(nparray1) * np.linalg.norm(nparray2))
            full_sim.append(float(cosine_similarity))
        return full_sim

    def _corpus_filter_poor_highlights(
        self, corpus_id: str, synth_questions_df: pd.DataFrame, threshold: float,
    ) -> None:
        corpus_questions_df = synth_questions_df[
            synth_questions_df['corpus_id'] == corpus_id].copy()

        def edit_row(row: pd.Series) -> pd.Series:
            question = row['question']
            references = [ref['content'] for ref in row['references']]
            similarity_scores = self._get_sim(question, references)
            row['worst_ref_score'] = min(similarity_scores)
            return row

        corpus_questions_df = corpus_questions_df.apply(edit_row, axis=1)
        count_before = len(corpus_questions_df)
        corpus_questions_df = corpus_questions_df[
            corpus_questions_df['worst_ref_score'] >= threshold]
        corpus_questions_df = corpus_questions_df.drop(columns=['worst_ref_score'])
        count_after = len(corpus_questions_df)
        print(f"Corpus: {corpus_id} - Removed {count_before - count_after} .")

        corpus_questions_df['references'] = corpus_questions_df['references'].apply(json.dumps)
        full_questions_df = pd.read_csv(self.questions_csv_path)
        full_questions_df = full_questions_df[
            full_questions_df['corpus_id'] != corpus_id]
        full_questions_df = pd.concat(
            [full_questions_df, corpus_questions_df], ignore_index=True)
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)
        full_questions_df.to_csv(self.questions_csv_path, index=False)

    def filter_poor_excerpts(
        self, threshold: float = 0.36, corpora_subset: Optional[List[str]] = None,
    ) -> None:
        if corpora_subset is None:
            corpora_subset = []
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                synth_questions_df['references'] = synth_questions_df['references'].apply(
                    json.loads)
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_poor_highlights(
                        corpus_id, synth_questions_df, threshold)

    def _corpus_filter_duplicates(
        self, corpus_id: str, synth_questions_df: pd.DataFrame, threshold: float,
    ) -> None:
        corpus_questions_df = synth_questions_df[
            synth_questions_df['corpus_id'] == corpus_id].copy()
        count_before = len(corpus_questions_df)
        corpus_questions_df.drop_duplicates(subset='question', keep='first', inplace=True)
        questions = corpus_questions_df['question'].tolist()

        response = self.client.embeddings.create(
            input=questions,
            model="text-embedding-3-large",
        )
        embeddings_matrix = np.array([data.embedding for data in response.data])
        dot_product_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)

        def filter_vectors(
            sim_matrix: np.ndarray, threshold: float,
        ) -> np.ndarray:
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
        full_questions_df = full_questions_df[
            full_questions_df['corpus_id'] != corpus_id]
        full_questions_df = pd.concat(
            [full_questions_df, corpus_questions_df], ignore_index=True)
        for col in ['fixed', 'worst_ref_score', 'diff_score']:
            if col in full_questions_df.columns:
                full_questions_df = full_questions_df.drop(columns=col)
        full_questions_df.to_csv(self.questions_csv_path, index=False)

    def filter_duplicates(
        self, threshold: float = 0.78, corpora_subset: Optional[List[str]] = None,
    ) -> None:
        if corpora_subset is None:
            corpora_subset = []
        if os.path.exists(self.questions_csv_path):
            synth_questions_df = pd.read_csv(self.questions_csv_path)
            if len(synth_questions_df) > 0:
                synth_questions_df['references'] = synth_questions_df['references'].apply(
                    json.loads)
                corpus_list = synth_questions_df['corpus_id'].unique().tolist()
                if corpora_subset:
                    corpus_list = [c for c in corpus_list if c in corpora_subset]
                for corpus_id in corpus_list:
                    self._corpus_filter_duplicates(
                        corpus_id, synth_questions_df, threshold)
