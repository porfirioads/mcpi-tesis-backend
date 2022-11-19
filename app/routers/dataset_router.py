from typing import List
from fastapi import APIRouter, File, UploadFile
from app.schemas.common_schemas import FileUpload
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.config import logger

router = APIRouter(prefix='/datasets', tags=['Datasets'])

dataset_service = DatasetService()
cleaning_service = CleaningService()


@router.get('/', response_model=List[FileUpload])
def get_dataset_list():
    logger.debug('get_dataset_list()')
    return dataset_service.get_datasets_list()


@router.post('/upload', response_model=FileUpload)
def upload_dataset(file: UploadFile = File()):
    logger.debug('upload_dataset()')
    return dataset_service.upload_dataset(file)


@router.get('/download')
def download_dataset(file_path: str):
    logger.debug('download_dataset()')
    return dataset_service.download_dataset(file_path)


@router.get('/clean', response_model=FileUpload)
def clean_dataset(file_path: str):
    logger.debug('clean_dataset()')
    return cleaning_service.clean(file_path)
