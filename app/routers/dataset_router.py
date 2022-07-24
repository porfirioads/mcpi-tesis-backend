from fastapi import APIRouter, Body, File, Form, UploadFile
from app.schemas.dataset_schemas import DatasetSummary
from app.services.dataset_service import DatasetService

router = APIRouter(prefix='/datasets', tags=['Datasets'])

dataset_service = DatasetService()


@router.get('')
def get_datasets_list():
    return ['22220723163026-tucson.csv', '22220722171824-encuesta.csv']


@router.post('/upload', response_model=DatasetSummary)
def upload_dataset(
    file: UploadFile = File(...),
    encoding: str = Form(...),
    delimiter: str = Form(..., example='|')
):
    file_path = dataset_service.upload_dataset(file)
    df = dataset_service.read_dataset(file_path, encoding, delimiter)
    return dataset_service.get_dataset_summary(df)
