from datetime import datetime
from fastapi import APIRouter
from app.schemas.common_schemas import FileUpload
from app.services.dataset_service import DatasetService
from app.services.pretrained_analysis_service import PretrainedAnalysisService
from app.config import logger

router = APIRouter(prefix='/analysis', tags=['Analysis'])

analysis_service = PretrainedAnalysisService()
dataset_service = DatasetService()


@router.get('/pretrained', response_model=FileUpload)
def pretrained(file_path: str):
    logger.debug('pretrained()')
    df = dataset_service.prepare_dataset(
        file_path=f'uploads/cleaned/{file_path}'
    )

    df_classified = analysis_service.classify_pretrained(
        df,
        'answer',
        'sentiment'
    )

    return dataset_service.to_csv(df_classified, file_path)


@router.get('/metrics')
def metrics(file_path: str):
    df = dataset_service.read_dataset(
        file_path=f'uploads/classified/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.get_metrics(
        df=df,
        y_true='sentiment',
        y_pred='max_voting'
    )
