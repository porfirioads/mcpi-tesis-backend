from app.utils.singleton import SingletonMeta
from textblob.classifiers import MaxEntClassifier, DecisionTreeClassifier, \
    NaiveBayesClassifier
import pandas as pd
from app.services.dataset_service import DatasetService
from app.config import logger


class TrainedAnalysisService(metaclass=SingletonMeta):
    dataset_service = DatasetService()

    def prepare_dataset(
        self,
        file_path: str,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        df = self.dataset_service.read_dataset(
            file_path=file_path,
            encoding='utf-8',
            delimiter=','
        )

        df['sentiment'] = df['sentiment'].replace(
            {
                'Positivo': 'pos',
                'Negativo': 'neg',
                'Neutral': 'neu'
            }
        )

        df = df[[text_column, target_column]]

        return df

    def train_models(self):
        logger.debug('Training DecisionTreeClassifier')
        self.dtcl = DecisionTreeClassifier(self.X_train.values)
        self.dtcl.train()
        logger.debug('Training MaxEntClassifier')
        self.mecl = MaxEntClassifier(self.X_train.values)
        self.mecl.train()
        logger.debug('Training NaiveBayesClassifier')
        self.nbcl = NaiveBayesClassifier(self.X_train.values)
        self.nbcl.train()
        logger.debug('Training finished')

    def classify_for_evaluation(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.dataset_service.split_dataset(df, 'sentiment')

        self.train_models()
        data = []
        i = 0  # Informative index in iteration

        for item in self.y_test.iteritems():
            logger.debug(f'classifying item {i + 1} of {len(self.y_test)}')
            text = df.iloc[item[0]][text_column]
            sentiment = df.iloc[item[0]][target_column]

            scores = [
                self.dtcl.classify(text),
                self.mecl.classify(text),
                self.nbcl.classify(text),
            ]

            data.append([
                text,
                sentiment,
                scores[0],
                scores[1],
                scores[2],
            ])

            i += 1

        new_df = pd.DataFrame(
            data,
            columns=[
                'answer',
                'sentiment',
                'decision_tree',
                'max_entrophy',
                'naive_bayes',
            ]
        )

        algorithms = [
            'sentiment',
            'decision_tree',
            'max_entrophy',
            'naive_bayes',
        ]

        for column in algorithms:
            new_df[column] = new_df[column].replace(
                {'pos': '1', 'neg': '-1', }
            )

        return new_df

    def classify(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.dataset_service.split_dataset(df, 'sentiment')
        self.train_models()
        data = []

        for index, row in df.iterrows():
            logger.debug(f'classifying item {index + 1} of {len(df)}')
            text = row[text_column]
            sentiment = row[target_column]

            scores = [
                self.dtcl.classify(text),
                self.mecl.classify(text),
                self.nbcl.classify(text),
            ]

            data.append([text, sentiment] + scores)

        new_df = pd.DataFrame(
            data,
            columns=[
                'answer',
                'sentiment',
                'decision_tree',
                'max_entrophy',
                'naive_bayes',
            ]
        )

        algorithms = [
            'sentiment',
            'decision_tree',
            'max_entrophy',
            'naive_bayes',
        ]

        for column in algorithms:
            new_df[column] = new_df[column].replace(
                {'pos': '1', 'neg': '-1', }
            )

        return new_df
