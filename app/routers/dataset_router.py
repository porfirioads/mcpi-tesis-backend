from typing import List
from fastapi import APIRouter, File, UploadFile
from app.schemas.common_schemas import FileUpload
from app.services.balance_service import BalanceService
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.config import logger

router = APIRouter(prefix='/datasets', tags=['Datasets'])

dataset_service = DatasetService()
cleaning_service = CleaningService()
balance_service = BalanceService()


@router.get('/', response_model=List[FileUpload])
def get_dataset_list():
    logger.debug('get_dataset_list()')
    return dataset_service.get_datasets_list(path='uploads')


@router.get('/cleaned', response_model=List[FileUpload])
def get_cleaned_dataset_list():
    logger.debug('get_cleaned_dataset_list()')
    return dataset_service.get_datasets_list(path='cleaned')


@router.get('/classified', response_model=List[FileUpload])
def get_classified_dataset_list():
    logger.debug('get_classified_dataset_list()')
    return dataset_service.get_datasets_list(path='classified')


@router.post('/upload', response_model=FileUpload)
def upload_dataset(file: UploadFile = File()):
    logger.debug('upload_dataset()')
    return dataset_service.upload_dataset(file)


@router.post('/download')
def download_dataset(file_path: str):
    logger.debug('download_dataset()')
    return dataset_service.download_dataset(file_path)


@router.post('/clean', response_model=FileUpload)
def clean_dataset(file_path: str):
    logger.debug('clean_dataset()')

    df_cleaned = cleaning_service.clean(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df_cleaned,
        file_path=f'resources/cleaned/cle_{file_path}'
    )


@router.post('/balance', response_model=FileUpload)
def balance_dataset(file_path: str):
    logger.debug('balance_dataset()')

    df_balanced = balance_service.balance(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df_balanced,
        file_path=f'resources/cleaned/bal_{file_path}'
    )


@router.post('/summary')
def summary_dataset(file_path: str):
    logger.debug('summary_dataset()')
    return dataset_service.summary_dataset(
        file_path=file_path,
        encoding='utf-8',
        delimiter=',',
        target_column='sentiment'
    )
