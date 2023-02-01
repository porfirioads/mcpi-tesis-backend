import pandas as pd
from typing import List
from app.patterns.singleton import SingletonMeta
from app.services.dataset_service import DatasetService
from app.services.nltk_service import NltkService
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

dataset_service = DatasetService()
nltk_service = NltkService()
columns = ['answer', 'sentiment']


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
            file_path=file_path,
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
        new_df = pd.DataFrame(data, columns=columns)

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
            file_path=file_path,
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
            synonym_phrase = nltk_service.extract_synonyms(answer)
            data.append([synonym_phrase, 'Negativo'])
            missing_negatives = missing_negatives - 1
            if not missing_negatives:
                break

        # Generate new dataset
        new_df = df.append(pd.DataFrame(data, columns=columns))

        return new_df

    def term_document_matrix(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        # Read original dataset
        df = dataset_service.read_dataset(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        # Clean the answers
        df['answer'] = df['answer']\
            .apply(nltk_service.to_lower)\
            .apply(nltk_service.remove_stop_words)\
            .apply(nltk_service.remove_punctuation)\
            .apply(nltk_service.remove_numbers)\
            .apply(nltk_service.stem_words)

        # Count Vectorizer
        vect = CountVectorizer()
        vects = vect.fit_transform(df.answer)

        # Term document matrix
        td = pd.DataFrame(vects.todense())
        td.columns = vect.get_feature_names_out()

        # Transpose dataframe
        term_document_matrix = td.T

        # Add column names
        term_document_matrix.columns = [
            answer for answer in df['answer'].values
        ]

        # Add total_count column
        term_document_matrix['total_count'] = term_document_matrix.sum(
            axis=1
        )

        # Sort by total_count column
        term_document_matrix = term_document_matrix.sort_values(
            by='total_count',
            ascending=False
        )

        # Keep words with al least 5 occurrences
        term_document_matrix = term_document_matrix[
            term_document_matrix['total_count'] >= 5
        ]

        # Transpose dataframe
        td = term_document_matrix.drop(columns=['total_count']).T

        # Replace frequency by a simple Yes or No value
        for column in td.columns.values:
            td[column] = np.where(td[column] >= 1, 'Yes', 'No')

        # Add answer and sentiment answer columns
        td.insert(loc=0, column='answer', value=td.index)
        td.insert(loc=1, column='sentiment', value=df['sentiment'].values)
        td = td.reset_index(drop=True)

        return td
