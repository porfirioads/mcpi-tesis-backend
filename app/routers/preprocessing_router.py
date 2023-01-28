from fastapi import APIRouter
from app.schemas.file_schemas import FileUpload
from app.services.dataset_service import DatasetService
from app.config import logger
from app.services.preprocessing_service import PreprocessingService

router = APIRouter(prefix='/preprocessing', tags=['Preprocessing'])

dataset_service = DatasetService()
preprocessing_service = PreprocessingService()


@router.post('/label', response_model=FileUpload)
def label_dataset(file_path: str):
    logger.debug('label_dataset()')

    df = preprocessing_service.label(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df,
        file_path=f'resources/labeled/lab_{file_path}'
    )


@router.post('/balance', response_model=FileUpload)
def balance_dataset(file_path: str):
    logger.debug('balance_dataset()')

    df = preprocessing_service.balance(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df,
        file_path=f'resources/labeled/bal_{file_path}'
    )


@router.post('/extract_features', response_model=FileUpload)
def extract_features_dataset(file_path: str):
    logger.debug('extract_features_dataset()')

    df = preprocessing_service.extract_features(
        file_path=file_path,
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df,
        file_path=f'resources/cleaned/cle_{file_path}'
    )
