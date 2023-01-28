from typing import List
from fastapi import APIRouter, File, UploadFile
from app.schemas.common_schemas import FileUpload
from app.services.file_service import FileService
from app.config import logger

router = APIRouter(prefix='/files', tags=['Files'])

file_service = FileService()


@router.get('/', response_model=List[FileUpload])
def get_dataset_list(path: str = 'uploads'):
    logger.debug('get_dataset_list()')
    return file_service.get_datasets_list(path=path)


@router.post('/upload', response_model=FileUpload)
def upload_dataset(file: UploadFile = File()):
    logger.debug('upload_dataset()')
    return file_service.upload_dataset(file)


@router.post('/download')
def download_dataset(file_path: str):
    logger.debug('download_dataset()')
    return file_service.download_dataset(file_path)
