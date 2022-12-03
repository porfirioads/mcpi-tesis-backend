from fastapi import APIRouter
from app.services.analysis_service import AnalysisService
from app.config import logger

router = APIRouter(prefix='/analysis', tags=['Analysis'])

analysis_service = AnalysisService()


@router.get('/pretrained')
def pretrained(file_path: str):
    logger.debug('pretrained()')
    df = analysis_service.prepare_dataset(file_path=file_path)
    df_classified = analysis_service.classify_pretrained(
        df,
        'answer',
        'sentiment'
    )
