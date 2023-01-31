from fastapi import APIRouter
from app.services.analysis_service import AnalysisService
from app.services.dataset_service import DatasetService
from app.config import logger

router = APIRouter(prefix='/analysis', tags=['Analysis'])

dataset_service = DatasetService()
analysis_service = AnalysisService()


@router.post('/naive_bayes_bernoulli')
def naive_bayes_bernoulli():
    logger.debug('naive_bayes_bernoulli()')

    result = analysis_service.naive_bayes(
        file_path=f'resources/original/tucson-caapcs-2021-05-13-20-45.csv',
        encoding='utf-8',
        delimiter=','
    )


@router.post('/logistic_regression')
def logistic_regression():
    logger.debug('logistic_regression()')

    result = analysis_service.logistic_regression(
        file_path=f'resources/original/tucson-caapcs-2021-05-13-20-45.csv',
        encoding='utf-8',
        delimiter=','
    )


@router.post('/svm')
def svm():
    logger.debug('svm()')

    result = analysis_service.svm(
        file_path=f'resources/original/tucson-caapcs-2021-05-13-20-45.csv',
        encoding='utf-8',
        delimiter=','
    )


@router.post('/assemble')
def assemble():
    logger.debug('svm()')

    result = analysis_service.assemble(
        file_path=f'resources/original/tucson-caapcs-2021-05-13-20-45.csv',
        encoding='utf-8',
        delimiter=','
    )
