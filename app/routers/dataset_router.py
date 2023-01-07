from fastapi import APIRouter
from app.schemas.common_schemas import FileUpload
from app.services.balance_service import BalanceService
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.config import logger

router = APIRouter(prefix='/datasets', tags=['Datasets'])

dataset_service = DatasetService()
cleaning_service = CleaningService()
balance_service = BalanceService()


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


@router.post('/extract_features', response_model=FileUpload)
def extract_features(file_path: str):
    logger.debug('extract_features()')

    df_extracted = cleaning_service.extract_features(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df_extracted,
        file_path=f'resources/cleaned/ext_{file_path}'
    )


@router.post('/extract_best_features', response_model=FileUpload)
def extract_best_features(file_path: str):
    logger.debug('extract_best_features()')

    df_extracted = cleaning_service.extract_best_features(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df_extracted,
        file_path=f'resources/cleaned/bst_{file_path}'
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
