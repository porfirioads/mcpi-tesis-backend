from app.services.dataset_service import DatasetService
from app.utils.singleton import SingletonMeta
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from app.config import logger
from app.services.cleaning_service import CleaningService

random_state = 100

classifiers = {
    "nearest_neighbors": KNeighborsClassifier(3),
    "linear_svm": SVC(
        kernel="linear",
        C=0.025,
        random_state=random_state,
        probability=True
    ),
    "rbf_svm": SVC(
        gamma=2,
        C=1,
        random_state=random_state,
        probability=True
    ),
    "gaussian_process": GaussianProcessClassifier(
        1.0 * RBF(1.0),
        random_state=random_state,
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=random_state
    ),
    "random_forest": RandomForestClassifier(
        max_depth=5,
        n_estimators=10,
        max_features=8,
        random_state=random_state
    ),
    "neural_net": MLPClassifier(
        alpha=1,
        max_iter=1000,
        random_state=random_state
    ),
    "ada_boost": AdaBoostClassifier(random_state=random_state),
    "naive_bayes": GaussianNB(),
    "qda": QuadraticDiscriminantAnalysis(),
    "gradient_boosting": GradientBoostingClassifier(
        random_state=random_state,
    ),
    "logistic_regression": LogisticRegression(random_state=random_state),
    "stochastic_gradient_descent": SGDClassifier(
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
        loss="modified_huber"
    )
}


class CustomAnalysisService(metaclass=SingletonMeta):
    dataset_service = DatasetService()
    cleaning_service = CleaningService()

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

        df[target_column] = df[target_column].replace(
            {
                'Positivo': '1',
                'Negativo': '-1',
                'Neutral': '0'
            }
        )

        return df

    def train_models(self):
        keys = list(classifiers.keys())
        for i in range(len(keys)):
            classifier = keys[i]
            classifiers[classifier].fit(self.X_train, self.y_train)
            logger.debug(
                f'Model {i + 1} of {len(classifiers)} '
                + f'({classifier}) trained.'
            )

    def classify(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        new_df = df.drop([text_column], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.dataset_service.split_dataset(new_df, target_column)

        self.train_models()
        data = []

        for index, row in df.iterrows():
            logger.debug(f'classifying item {index + 1} of {len(df)}')
            text = row[text_column]
            sentiment = row[target_column]
            fields = self.cleaning_service.calculate_data_fields(text)
            del fields[0]
            fields = [sentiment] + fields
            fields = [int(field) for field in fields]
            fields = np.array(fields).reshape(1, -1)

            scores = [
                classifiers['nearest_neighbors'].predict(fields)[0],
                classifiers['linear_svm'].predict(fields)[0],
                classifiers['rbf_svm'].predict(fields)[0],
                classifiers['gaussian_process'].predict(fields)[0],
                classifiers['decision_tree'].predict(fields)[0],
                classifiers['random_forest'].predict(fields)[0],
                classifiers['neural_net'].predict(fields)[0],
                classifiers['ada_boost'].predict(fields)[0],
                classifiers['naive_bayes'].predict(fields)[0],
                classifiers['qda'].predict(fields)[0],
                classifiers['gradient_boosting'].predict(fields)[0],
                classifiers['logistic_regression'].predict(fields)[0],
                classifiers['stochastic_gradient_descent'].predict(fields)[0],
            ]

            data.append([
                text,
                sentiment,
                scores[0],
                scores[1],
                scores[2],
                scores[3],
                scores[4],
                scores[5],
                scores[6],
                scores[7],
                scores[8],
                scores[9],
                scores[10],
                scores[11],
                scores[12],
                self.dataset_service.most_frequent(scores)
            ])

        return pd.DataFrame(
            data,
            columns=[
                'answer',
                'sentiment',
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
                # 'stochastic_gradient_descent',
            ]
        )

    def classify_for_evaluation(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        new_df = df.drop([text_column], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.dataset_service.split_dataset(new_df, target_column)

        self.train_models()

        algorithms = [
            'linear_svm',
            'nearest_neighbors',
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
            # 'stochastic_gradient_descent',
        ]

        probas = [f'{algorithm}_proba' for algorithm in algorithms]

        new_df = pd.DataFrame(
            [],
            columns=['answer', 'sentiment'] + algorithms + probas
        )

        answers = []
        sentiments = []

        for item in self.y_test.iteritems():
            answers.append(df.iloc[item[0]][text_column])
            sentiments.append(df.iloc[item[0]][target_column])

        new_df['answer'] = answers
        new_df['sentiment'] = sentiments

        for algorithm in algorithms:
            logger.debug(f'Executing algorithm {algorithm}')
            result = classifiers[algorithm].predict(self.X_test)
            result_proba = classifiers[algorithm].predict_proba(self.X_test)
            new_df[algorithm] = result
            new_df[f'{algorithm}_proba'] = result_proba

        return new_df
