from typing import Any, Dict, List
from pydantic import BaseModel


class DatasetSummary(BaseModel):
    row_count: int
    col_count: int
    columns: List[str]
    samples: Dict[str, Any]
