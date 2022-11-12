from typing import List
from fastapi import APIRouter, File, Form, UploadFile
from app.schemas.common_schemas import FileUpload
from app.schemas.dataset_schemas import DatasetCleanInput, DatasetSummary
from fastapi.responses import FileResponse
from app.services.dataset_service import DatasetService
from app.utils import datasets

router = APIRouter(prefix='/preprocessor', tags=['Preprocessor'])

dataset_service = DatasetService()


@router.get('/datasets/', response_model=List[str])
def get_datasets_list():
    return dataset_service.get_datasets_list()


@router.post('/datasets/upload', response_model=FileUpload)
def upload_dataset(
    file: UploadFile = File(),
):
    file_path = dataset_service.upload_dataset(file)
    return FileUpload(file_path=file_path)
