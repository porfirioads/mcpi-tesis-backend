from typing import List
from fastapi import APIRouter, File, Form, UploadFile
from app.schemas.common_schemas import FileUpload
from app.schemas.dataset_schemas import DatasetCleanInput, DatasetSummary
from fastapi.responses import FileResponse
from app.services.preprocessor_service import PreprocessorService
from app.utils import datasets

router = APIRouter(prefix='/preprocessor', tags=['Preprocessor'])

preprocessor_service = PreprocessorService()


@router.post('/upload', response_model=FileUpload)
def upload_dataset(
    file: UploadFile = File(),
):
    file_path = preprocessor_service.upload_dataset(file)
    return FileUpload(file_path=file_path)


@router.get('', response_model=List[str])
def get_datasets_list():
    return dataset_service.get_datasets_list()
