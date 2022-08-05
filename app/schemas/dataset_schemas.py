from typing import Any, Dict, List
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    row_count: int
    col_count: int
    columns: List[str]
    samples: Dict[str, Any]


class DatasetCleanInput(BaseModel):
    file_path: str = Field(..., example='1659667852_respuestas_tucson.csv')
    encoding: str = Field(..., example='latin-1')
    delimiter: str = Field(..., example='|')
    selected_columns: List[str] = Field(..., example=[
        # 'question_number',
        'sentiment',
        'answer',
        'length',
        'cc',
        'vbp',
        'cd',
        'dt',
        'ex',
        'fw',
        'inn',
        'jj',
        'jjr',
        'jjs',
        'ls',
        'md',
        'nn',
        'nns',
        'nnp',
        'nnps',
        'pdt',
        'posss',
        'prp',
        'rb',
        'rbr',
        'rp',
        'to',
        'uh',
        'vb',
        'vbd',
        'vbg',
        'vbn',
        'vbz',
        'wdt',
        'wp',
        'wps',
        'wrb',
        'pos_length',
        'neg_length',
        'pos_acumulado',
        'neg_acumulado',
        'polarity',
        'cant_caracteres'
    ])
    target_column: str = Field(..., example='Sentiment')
    text_column: str = Field(..., example='Answer')
    generated_columns: List[str] = Field(..., example=['cc', 'vbp', 'cd'])
