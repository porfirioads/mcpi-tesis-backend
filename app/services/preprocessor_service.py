from http.client import HTTPException
from fastapi import UploadFile, HTTPException
from app.schemas.common_schemas import FileUpload
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
from datetime import datetime


class PreprocessorService(metaclass=SingletonMeta):
    def upload_dataset(self, file: UploadFile):
        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_name = f'uploads/{timestamp}_{snack_name}'

        with open(file_name, 'wb') as f:
            f.write(file.file.read())

        return file_name
