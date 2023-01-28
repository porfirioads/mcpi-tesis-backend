from fastapi import APIRouter
from app.schemas.file_schemas import FileUpload
from app.services.dataset_service import DatasetService
from app.config import logger

router = APIRouter(prefix='/training', tags=['Analysis'])

dataset_service = DatasetService()


@router.post('/naive_bayes_bernoulli', response_model=FileUpload)
def train_naive_bayes_bernoulli(file_path: str):
    logger.debug('train_naive_bayes_bernoulli()')


@router.post('/logistic_regression', response_model=FileUpload)
def train_logistic_regression(file_path: str):
    logger.debug('train_logistic_regression()')


@router.post('/svm', response_model=FileUpload)
def train_svm(file_path: str):
    logger.debug('train_svm()')
