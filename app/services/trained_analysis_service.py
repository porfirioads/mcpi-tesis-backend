from app.utils.singleton import SingletonMeta
from textblob.classifiers import MaxEntClassifier, DecisionTreeClassifier, NaiveBayesClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from app.services.dataset_service import DatasetService


class TrainedAnalysisService(metaclass=SingletonMeta):
    dataset_service = DatasetService()

    def prepare_dataset(self, file_path: str) -> pd.DataFrame:
        df = self.dataset_service.read_dataset(
            file_path=file_path,
            encoding='utf-8',
            delimiter='|'
        )

        df['sentiment'] = df['sentiment'].replace(
            {
                'Positivo': 'pos',
                'Negativo': 'neg',
                'Neutral': 'neu'
            }
        )

        return df

    def split_dataset(self, df: pd.DataFrame, y_true: str):
        y = df[y_true]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df,
            y,
            test_size=0.25,
            random_state=1
        )

    def train_models(self):
        self.dtcl = DecisionTreeClassifier(self.X_train.values)
        self.dtcl.train()
        self.mecl = MaxEntClassifier(self.X_train.values)
        self.mecl.train()
        self.nbcl = NaiveBayesClassifier(self.X_train.values)
        self.nbcl.train()

    def classify(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        new_df = df[[text_column, target_column]]
        self.split_dataset(new_df, 'sentiment')
        self.train_models()
        data = []

        for index, row in new_df.iterrows():
            print(f'classifying item {index + 1} of {len(new_df)}')
            text = row[text_column]
            sentiment = row[target_column]

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
                self.dataset_service.most_frequent(scores)
            ])

        return pd.DataFrame(
            data,
            columns=[
                'answer',
                'sentiment',
                'decision_tree',
                'max_entrophy',
                'naive_bayes',
                'max_voting'
            ]
        )
