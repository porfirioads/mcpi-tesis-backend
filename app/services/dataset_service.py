import os
from typing import List
from fastapi import UploadFile, HTTPException
from app.schemas.common_schemas import FileUpload
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
from fastapi.responses import FileResponse
from datetime import datetime
from app.utils import datasets


class DatasetService(metaclass=SingletonMeta):
    def upload_dataset(self, file: UploadFile) -> str:
        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_path = f'{timestamp}_{snack_name}'

        with open(f'uploads/{file_path}', 'wb') as f:
            f.write(file.file.read())

        return FileUpload(file_path=file_path)

    def get_datasets_list(self, path: str = None) -> List[str]:
        files = os.listdir(f'uploads/{path}' if path else 'uploads')

        if '.gitkeep' in files:
            files.remove('.gitkeep')

        if 'cleaned' in files:
            files.remove('cleaned')

        return [
            FileUpload(file_path=f'{file_name}') for file_name in files
        ]

    def download_dataset(self, file_path: str) -> FileResponse:
        os.stat(f'uploads/{file_path}')
        return FileResponse(
            path=f'uploads/{file_path}',
            media_type='text/csv',
            filename=file_path
        )

    def summary_dataset(self, file_path: str, encoding: str, delimiter) -> dict:
        os.stat(f'uploads/cleaned/{file_path}')
        df = datasets.read_dataset(
            file_path=f'uploads/cleaned/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )
        return df['sentiment'].value_counts().to_dict()
