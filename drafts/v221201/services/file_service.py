import os
from app.config import logger
from app.schemas.common_schemas import FileUpload
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
from typing import List
from fastapi import UploadFile, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime


class FileService(metaclass=SingletonMeta):
    def upload_dataset(self, file: UploadFile) -> FileUpload:
        logger.debug('DatasetService.upload_dataset()')

        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_path = f'{snack_name[0: -4]}_{timestamp}.csv'

        with open(f'resources/uploads/{file_path}', 'wb') as f:
            f.write(file.file.read())

        return FileUpload(file_path=file_path)

    def get_datasets_list(self, path: str) -> List[FileUpload]:
        logger.debug('DatasetService.get_datasets_list()')
        files = os.listdir(f'resources/{path}')

        if '.gitkeep' in files:
            files.remove('.gitkeep')

        if 'cleaned' in files:
            files.remove('cleaned')

        if 'classified' in files:
            files.remove('classified')

        return [
            FileUpload(file_path=f'{file_name}') for file_name in files
        ]

    def download_dataset(self, file_path: str) -> FileResponse:
        logger.debug('DatasetService.download_dataset()')
        os.stat(f'resources/{file_path}')

        return FileResponse(
            path=f'resources/{file_path}',
            media_type='text/csv',
            filename=file_path
        )
