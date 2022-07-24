from datetime import datetime
from typing import List
from fastapi import UploadFile, HTTPException
from app.schemas.dataset_schemas import DatasetSummary
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
import json
import pandas as pd
import os


class DatasetService(metaclass=SingletonMeta):
    def get_datasets_list(self) -> List[str]:
        return os.listdir('uploads')

    def upload_dataset(self, file: UploadFile) -> str:
        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_path = f'uploads/{timestamp}_{snack_name}'

        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        return file_path

    def read_dataset(self, file_path: str, encoding: str, delimiter: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        return df

    def get_dataset_summary(self, df: pd.DataFrame) -> DatasetSummary:
        return DatasetSummary(
            row_count=df.shape[0],
            col_count=df.shape[1],
            columns=list(df.columns.values),
            samples=json.loads(df.head(5).transpose().to_json())
        )
