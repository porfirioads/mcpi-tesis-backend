from datetime import datetime
from fastapi import APIRouter
from app.schemas.common_schemas import FileUpload
from app.services.analysis_service import AnalysisService
from app.config import logger

router = APIRouter(prefix='/analysis', tags=['Analysis'])

analysis_service = AnalysisService()


@router.get('/pretrained', response_model=FileUpload)
def pretrained(file_path: str):
    logger.debug('pretrained()')
    df = analysis_service.prepare_dataset(file_path=file_path)

    df_classified = analysis_service.classify_pretrained(
        df,
        'answer',
        'sentiment'
    )

    timestamp = int(datetime.timestamp(datetime.now()))
    file_path = f'{file_path[0: -4]}_{timestamp}.csv'

    df_classified.to_csv(
        f'uploads/classified/{file_path}',
        index=False
    )

    return FileUpload(file_path=file_path)
