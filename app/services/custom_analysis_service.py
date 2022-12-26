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

    def classify_for_evaluation(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        logger.debug('CustomAnalysisService.classify_for_evaluation() start')
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

        for item in self.y_test.items():
            answers.append(df.iloc[item[0]][text_column])
            sentiments.append(df.iloc[item[0]][target_column])

        new_df['answer'] = answers
        new_df['sentiment'] = sentiments

        for algorithm in algorithms:
            result = classifiers[algorithm].predict(self.X_test)
            result_proba = classifiers[algorithm].predict_proba(self.X_test)
            result_proba = [str(proba) for proba in result_proba]
            new_df[algorithm] = result
            new_df[f'{algorithm}_proba'] = result_proba

        logger.debug('CustomAnalysisService.classify_for_evaluation() end')
        return new_df
