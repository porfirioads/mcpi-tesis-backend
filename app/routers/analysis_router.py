from fastapi import APIRouter
from app.schemas.common_schemas import FileUpload
from app.services.dataset_service import DatasetService
from app.services.pretrained_analysis_service import PretrainedAnalysisService
from app.config import logger
from app.services.trained_analysis_service import TrainedAnalysisService
from app.services.custom_analysis_service import CustomAnalysisService

router = APIRouter(prefix='/analysis', tags=['Analysis'])

pretrained_analysis_service = PretrainedAnalysisService()
trained_analysis_service = TrainedAnalysisService()
custom_analysis_service = CustomAnalysisService()
dataset_service = DatasetService()


@router.post('/pretrained', response_model=FileUpload)
def pretrained(file_path: str):
    logger.debug('pretrained()')

    df = pretrained_analysis_service.prepare_dataset(
        file_path=f'resources/cleaned/{file_path}',
        text_column='answer',
        target_column='sentiment'
    )

    df_classified = pretrained_analysis_service.classify(
        df,
        'answer',
        'sentiment'
    )

    return dataset_service.to_csv(
        df_classified,
        f'resources/classified/pretrained_{file_path}'
    )


@router.post('/trained_evaluation', response_model=FileUpload)
def trained(file_path: str):
    logger.debug('trained()')

    df = trained_analysis_service.prepare_dataset(
        file_path=f'resources/cleaned/{file_path}',
        text_column='answer',
        target_column='sentiment'
    )

    df_classified = trained_analysis_service.classify_for_evaluation(
        df,
        'answer',
        'sentiment'
    )

    return dataset_service.to_csv(
        df_classified,
        f'resources/classified/trained_evaluation_{file_path}'
    )


@router.post('/trained', response_model=FileUpload)
def trained(file_path: str):
    logger.debug('trained()')

    df = trained_analysis_service.prepare_dataset(
        file_path=f'resources/cleaned/{file_path}',
        text_column='answer',
        target_column='sentiment'
    )

    df_classified = trained_analysis_service.classify(
        df,
        'answer',
        'sentiment'
    )

    return dataset_service.to_csv(
        df_classified,
        f'resources/classified/trained_evaluation_{file_path}'
    )


@router.post('/custom', response_model=FileUpload)
def custom(file_path: str):
    logger.debug('custom()')

    df = custom_analysis_service.prepare_dataset(
        file_path=f'resources/cleaned/{file_path}',
        text_column='answer',
        target_column='sentiment'
    )

    df_classified = custom_analysis_service.classify_for_evaluation(
        df,
        'answer',
        'sentiment'
    )

    return dataset_service.to_csv(
        df_classified,
        f'resources/classified/custom_{file_path}'
    )


@router.post('/custom_evaluation', response_model=FileUpload)
def custom(file_path: str):
    logger.debug('custom()')

    df = custom_analysis_service.prepare_dataset(
        file_path=f'resources/cleaned/{file_path}',
        text_column='answer',
        target_column='sentiment'
    )

    df_classified = custom_analysis_service.classify(
        df,
        'answer',
        'sentiment'
    )

    return dataset_service.to_csv(
        df_classified,
        f'resources/classified/custom_{file_path}'
    )


@router.post('/metrics')
def metrics(file_path: str):
    df = dataset_service.read_dataset(
        file_path=f'resources/classified/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    return dataset_service.get_metrics(
        df=df,
        y_true='sentiment',
        y_pred='max_voting'
    )
