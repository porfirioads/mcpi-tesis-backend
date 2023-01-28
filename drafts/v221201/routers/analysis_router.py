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
        f'resources/classified/pre_all_{file_path}'
    )


@router.post('/trained_evaluation', response_model=FileUpload)
def trained_evaluation(file_path: str):
    logger.debug('trained_evaluation()')

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
        f'resources/classified/tra_eva_{file_path}'
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
        f'resources/classified/tra_all_{file_path}'
    )


@router.post('/custom', response_model=FileUpload)
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
        f'resources/classified/cus_all_{file_path}'
    )


@router.post('/custom_evaluation', response_model=FileUpload)
def custom_evaluation(file_path: str):
    logger.debug('custom_evaluation()')

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
        f'resources/classified/cus_eva_{file_path}'
    )


@router.post('/metrics_pretrained')
def metrics_pretrained(file_path: str):
    df = dataset_service.read_dataset(
        file_path=f'resources/classified/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    classifiers = [
        'vader',
        'textblob',
        'naive_bayes',
    ]

    metrics = {}

    for classifier in classifiers:
        metrics[classifier] = dataset_service.get_metrics(
            df=df,
            y_true='sentiment',
            y_pred=classifier,
            has_probability=False
        )

    metrics_df = dataset_service.metrics_to_df(metrics=metrics)

    return dataset_service.to_csv(
        metrics_df,
        f'resources/metrics/met_{file_path}'
    )


@router.post('/metrics_trained')
def metrics_trained(file_path: str):
    df = dataset_service.read_dataset(
        file_path=f'resources/classified/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    classifiers = [
        'decision_tree',
        'max_entrophy',
        'naive_bayes',
    ]

    metrics = {}

    for classifier in classifiers:
        metrics[classifier] = dataset_service.get_metrics(
            df=df,
            y_true='sentiment',
            y_pred=classifier,
            has_probability=False
        )

    metrics_df = dataset_service.metrics_to_df(metrics=metrics)

    return dataset_service.to_csv(
        metrics_df,
        f'resources/metrics/met_{file_path}'
    )


@router.post('/metrics_custom')
def metrics_custom(file_path: str):
    df = dataset_service.read_dataset(
        file_path=f'resources/classified/{file_path}',
        encoding='utf-8',
        delimiter=','
    )

    classifiers = [
        'nearest_neighbors',
        'linear_svm',
        'rbf_svm',
        'gaussian_process',
        'decision_tree',
        'random_forest',
        'neural_net',
        'ada_boost',
        'naive_bayes',
        'qda',
        'gradient_boosting',
        'logistic_regression',
    ]

    metrics = {}

    for classifier in classifiers:
        metrics[classifier] = dataset_service.get_metrics(
            df=df,
            y_true='sentiment',
            y_pred=classifier,
            has_probability=True
        )

    metrics_df = dataset_service.metrics_to_df(metrics=metrics)

    return dataset_service.to_csv(
        metrics_df,
        f'resources/metrics/met_{file_path}'
    )
