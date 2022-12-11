from typing import List
from app.schemas.common_schemas import FileUpload
from app.services.cleaning_service import CleaningService
from app.services.dataset_service import DatasetService
from app.services.phrase_replacer_service import PhraseReplacerService
from app.utils.singleton import SingletonMeta
from app.config import logger
from datetime import datetime
import pandas as pd


class BalanceService(metaclass=SingletonMeta):
    dataset_service = DatasetService()
    phrase_replacer_service = PhraseReplacerService()
    cleaning_service = CleaningService()

    def join_categories(
        self,
        df: pd.DataFrame,
        source_column: str,
        categories_to_join: List[str],
        target_category: str
    ):
        replacements = {}

        for category in categories_to_join:
            replacements[category] = target_category

        df[source_column] = df[source_column].replace(replacements)

        return df

    def balance(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> FileUpload:
        # Read source dataset.
        df = self.dataset_service.read_dataset(
            file_path=f'resources/cleaned/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        # Negative and neutral answers are now considered as Nega
        df = self.join_categories(
            df=df,
            source_column='sentiment',
            categories_to_join=['Neutral', 'Negativo'],
            target_category='Negativo'
        )

        # Calculate how many negative answers are necessary to get a balanced
        # dataset
        frequencies = df['sentiment'].value_counts().to_dict()
        missing_negatives = frequencies['Positivo'] - frequencies['Negativo']
        negatives = df[df['sentiment'] == 'Negativo']

        for negative, index in negatives.iterrows():
            synonym_phrase = self.phrase_replacer_service.extract_synonyms(
                negative
            )

            logger.debug(synonym_phrase)

            # TODO: Add phrase to the dataset
            missing_negatives = missing_negatives - 1

            if not missing_negatives:
                break

        # New dataset file name
        timestamp = int(datetime.timestamp(datetime.now()))
        file_path = f'{file_path[0: -4]}_{timestamp}.csv'
