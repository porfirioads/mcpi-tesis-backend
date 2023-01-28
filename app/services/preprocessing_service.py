import pandas as pd
from typing import List
from app.patterns.singleton import SingletonMeta
from app.services.dataset_service import DatasetService
from app.services.phrase_replacer_service import PhraseReplacerService

dataset_service = DatasetService()
phrase_replacer_service = PhraseReplacerService()


class PreprocessingService(metaclass=SingletonMeta):
    def join_categories(
        self,
        df: pd.DataFrame,
        source_column: str,
        categories_to_join: List[str],
        target_category: str
    ) -> pd.DataFrame:
        replacements = {}

        for category in categories_to_join:
            replacements[category] = target_category

        df[source_column] = df[source_column].replace(replacements)

        return df

    def label(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        # Read original dataset
        df = dataset_service.read_dataset(
            file_path=f'resources/uploads/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        # Get the columns that contains answers
        answers = df.columns[3:].to_list()

        # Variable where rows will be stored
        data = []

        # Iterate answers to generate the new dataset
        for answer in answers:
            sentiment = df[answer].value_counts().idxmax()
            data.append([answer, sentiment])

        # Generate new dataset
        new_df = pd.DataFrame(
            data,
            columns=['answer', 'sentiment']
        )

        # Negative and neutral answers are now considered as negative
        new_df = self.join_categories(
            df=new_df,
            source_column='sentiment',
            categories_to_join=['Neutral', 'Negativo'],
            target_category='Negativo'
        )

        return new_df

    def balance(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        # Read original dataset
        df = dataset_service.read_dataset(
            file_path=f'resources/labeled/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        # Calculate how many negative answers are necessary to get a balanced
        # dataset
        frequencies = df['sentiment'].value_counts().to_dict()
        missing_negatives = frequencies['Positivo'] - frequencies['Negativo']
        negatives = df[df['sentiment'] == 'Negativo']

        # Variable where rows will be stored
        data = []

        for index, row in negatives.iterrows():
            answer = row['answer']
            synonym_phrase = phrase_replacer_service.extract_synonyms(answer)
            data.append([synonym_phrase, 'Negativo'])
            missing_negatives = missing_negatives - 1
            if not missing_negatives:
                break

        # Generate new dataset
        new_df = df.append(pd.DataFrame(data, columns=['answer', 'sentiment']))

        return new_df

    def extract_features(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        pass
