import os
from pathlib import Path

from .base_evaluation import BaseEvaluation

_GENERAL_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'general_evaluation_data')
_GENERAL_DATA_DIR = os.path.abspath(_GENERAL_DATA_DIR)


class GeneralEvaluation(BaseEvaluation):
    def __init__(self, chroma_db_path=None):
        questions_df_path = os.path.join(_GENERAL_DATA_DIR, 'questions_df.csv')
        corpora_folder = Path(os.path.join(_GENERAL_DATA_DIR, 'corpora'))
        corpora_filenames = [f for f in corpora_folder.iterdir() if f.is_file()]
        corpora_id_paths = {f.stem: str(f) for f in corpora_filenames}

        super().__init__(str(questions_df_path), chroma_db_path=chroma_db_path,
                         corpora_id_paths=corpora_id_paths)
        self.is_general = True
