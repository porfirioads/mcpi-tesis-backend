from typing import List
from fastapi import APIRouter, File, Form, UploadFile
from app.schemas.dataset_schemas import DatasetCleanInput, DatasetSummary
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.utils import datasets

router = APIRouter(prefix='/datasets', tags=['Datasets'])

dataset_service = DatasetService()
cleaning_service = CleaningService()


@router.get('', response_model=List[str])
def get_datasets_list():
    return dataset_service.get_datasets_list()


@router.get('/{file_path}', response_model=DatasetSummary)
def get_dataset(
        file_path: str,
        encoding: str = 'latin-1',
        delimiter: str = ','
):
    df = datasets.read_dataset(f'uploads/{file_path}', encoding, delimiter)
    return dataset_service.get_dataset_summary(df)


@router.post('/upload', response_model=DatasetSummary)
def upload_dataset(
    file: UploadFile = File(),
    encoding: str = Form('latin-1'),
    delimiter: str = Form(',')
):
    file_path = dataset_service.upload_dataset(file)
    df = datasets.read_dataset(file_path, encoding, delimiter)
    return dataset_service.get_dataset_summary(df)


@router.post('/clean', response_model=DatasetSummary, responses={})
def clean_dataset(data: DatasetCleanInput):
    file_path = cleaning_service.clean_dataset(data)
    df = datasets.read_dataset(file_path, data.encoding, data.delimiter)
    return dataset_service.get_dataset_summary(df)
