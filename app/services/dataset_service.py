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
        datasets = os.listdir('uploads')
        # datasets.remove('.gitkeep')
        return datasets

    def upload_dataset(self, file: UploadFile) -> str:
        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_name = f'uploads/{timestamp}_{snack_name}'

        with open(file_name, 'wb') as f:
            f.write(file.file.read())

        return file_name

    def get_dataset_summary(self, df: pd.DataFrame) -> DatasetSummary:
        return DatasetSummary(
            row_count=df.shape[0],
            col_count=df.shape[1],
        )
