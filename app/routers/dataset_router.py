from datetime import datetime
from fastapi import APIRouter, Body, File, Form, HTTPException, Path, UploadFile
from app.schemas.exploration_schemas import Exploration
from app.services.dataset_service import DatasetService
from app.utils.strings import Strings


router = APIRouter(prefix='/datasets', tags=['Datasets'])

dataset_service = DatasetService()


@router.get('')
async def get_datasets_list():
    return ['22220723163026-tucson.csv', '22220722171824-encuesta.csv']


@router.post('/upload')
async def upload_dataset(
    file: UploadFile = File(...),
    encoding: str = Form(...),
    delimiter: str = Form(...)
):
    uploaded_file_path = dataset_service.upload_dataset(file)
    dataset_summary = dataset_service.read_csv(
        uploaded_file_path,
        encoding,
        delimiter
    )
    return uploaded_file_path
