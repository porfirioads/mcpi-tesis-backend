from typing import Any, Dict, List
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    row_count: int
    col_count: int
    columns: List[str]
    samples: Dict[str, Any]


class DatasetCleanInput(BaseModel):
    dataset_name: str = Field(..., example='1658682278_tucson.csv')
    selected_columns: List[str] = Field(..., example=['text', 'polarity', 'words'])
    target_column: str = Field(..., example='sentiment')
    text_column: str = Field(..., example="text")
    generated_columns: List[str] = Field(..., example=['cc', 'vbp', 'cd'])
