import re
import joblib
from app.patterns.singleton import SingletonMeta
from app.services.dataset_service import DatasetService
from app.services.nltk_service import NltkService
from app.services.wordcloud_service import WordcloudService
import pandas as pd

dataset_service = DatasetService()
nltk_service = NltkService()
wordcloud_service = WordcloudService()

questions = {
    'Q_ADAPTATION': 'Do you have other climate adaptation ideas? Submit here:',
    'Q_MITIGATION': 'Do you have other climate mitigation ideas? Submit here:',
    'Q_EQUITY': 'Do you have other ideas for environmental equity, justice, and community resilience? Submit here:',
    'Q_POLICY': 'Do you have other policy ideas? Submit here:',
    'Q_SUSTAINABLE': 'Are you interested in participating in any other ways to help make Tucson environmentally sustainable? Submit here:',
    'Q_OTHER': 'Is there anything else you would like to share that was not already addressed?'
}


class AnalysisService(metaclass=SingletonMeta):
    def execute_algorithm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str,
        model_file_path: str
    ):
        df = dataset_service.read_dataset(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
        )

        model = joblib.load(model_file_path)

        for question in questions:
            q_df = pd.DataFrame()
            q_df['answer'] = df[questions[question]].values.tolist()
            q_df = q_df.replace(r'\n', ' ', regex=True)
            q_df = q_df.replace('(?i)tucson', '', regex=True)
            q_df = q_df.replace('(?i)city', '', regex=True)
            q_df = q_df.replace('(?i)city', '', regex=True)
            q_df = q_df.replace(r'^\s*no\s*$', '', regex=True)
            q_df = q_df.replace(r'^\s*NO\s*$', '', regex=True)
            q_df = q_df.replace(r'^\s*No\s*$', '', regex=True)
            q_df = q_df.replace(r'^\s*nO\s*$', '', regex=True)
            q_df = q_df.replace(r'^\s*-\s*$', '', regex=True)
            nan_value = float("NaN")
            q_df.replace('', nan_value, inplace=True)
            q_df = q_df.dropna()
            text = ', '.join(str(x) for x in q_df['answer'].values.tolist())

            wordcloud_service.generate_wordcloud(
                text=text,
                title=questions[question],
                file_path=f'resources/wordclouds/{question}.png'
            )

            # model.predict(opinion)

    def logistic_regression(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        return self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model_file_path='resources/models/lgr.sav'
        )

    def naive_bayes(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        return self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model_file_path='resources/models/nbb.sav'
        )

    def svm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        return self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model_file_path='resources/models/svm.sav'
        )

    def assemble(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        self.logistic_regression(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        self.naive_bayes(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        self.svm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )
