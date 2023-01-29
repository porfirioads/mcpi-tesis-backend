from typing import Optional
from pydantic import BaseModel
import pandas as pd


class AlgorithmMetrics(BaseModel):
    accuracy: float
    f1: float
    precision: float
    sensitivity: float
    specificity: float


class AlgorithmResult(BaseModel):
    df: pd.DataFrame
    auroc_file_path: Optional[str]
    confusion_file_path: str
    metrics_file_path: Optional[str]
    metrics: AlgorithmMetrics

    class Config:
        arbitrary_types_allowed = True
