from fastapi import APIRouter
from app.schemas.file_schemas import FileUpload
from app.services.dataset_service import DatasetService
from app.config import logger
from app.services.preprocessing_service import PreprocessingService

router = APIRouter(prefix='/preprocessing', tags=['Preprocessing'])

dataset_service = DatasetService()
preprocessing_service = PreprocessingService()


@router.get('/details/{file_path}')
def get_dataset(
        file_path: str,
        encoding: str = 'latin-1',
        delimiter: str = ','
):
    df = dataset_service.read_dataset(
        file_path=f'resources/uploads/{file_path}',
        encoding=encoding,
        delimiter=delimiter
    )
    return dataset_service.get_dataset_summary(df)


@router.post('/label', response_model=FileUpload)
def label_dataset(file_path: str):
    logger.debug('label_dataset()')

    df = preprocessing_service.label(
        file_path=f'resources/uploads/{file_path}',
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
        file_path=f'resources/labeled/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df,
        file_path=f'resources/labeled/bal_{file_path}'
    )


@router.post('/term_document_matrix', response_model=FileUpload)
def generate_term_document_matrix(file_path: str):
    logger.debug('generate_term_document_matrix()')

    df = preprocessing_service.term_document_matrix(
        file_path=f'resources/labeled/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.to_csv(
        df=df,
        file_path=f'resources/cleaned/tdm_{file_path}'
    )
