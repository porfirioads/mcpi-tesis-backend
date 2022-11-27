from fastapi import APIRouter
from app.services.analysis_service import AnalysisService
from app.config import logger

router = APIRouter(prefix='/datasets', tags=['Analysis'])

analysis_service = AnalysisService()


@router.get('/')
def vader(file_path: str):
    logger.debug('vader()')
    return analysis_service.vader(file_path)
