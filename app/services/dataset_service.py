import os
from typing import List
from fastapi import UploadFile, HTTPException
from app.schemas.common_schemas import FileUpload
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
from datetime import datetime


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

    def get_datasets_list(self) -> List[str]:
        files = os.listdir('uploads')
        files.remove('.gitkeep')
        return [FileUpload(file_path=f'{file_name}') for file_name in files]
