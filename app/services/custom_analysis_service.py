from app.services.dataset_service import DatasetService
from app.utils.singleton import SingletonMeta
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean
from sklearn import model_selection, preprocessing
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from app.config import logger

random_state = 100

classifiers = {
    "nearest_neighbors": KNeighborsClassifier(3),
    "linear_svm": SVC(kernel="linear", C=0.025, random_state=random_state),
    "rbf_svm": SVC(gamma=2, C=1, random_state=random_state),
    "gaussian_process": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=random_state),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=random_state),
    "random_forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=8, random_state=random_state),
    "neural_net": MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
    "ada_boost": AdaBoostClassifier(random_state=random_state),
    "naive_bayes": GaussianNB(),
    "qda": QuadraticDiscriminantAnalysis(),
    "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    "logistic_regression": LogisticRegression(random_state=random_state),
    "stochastic_gradient_descent": SGDClassifier(max_iter=10000, tol=1e-3, random_state=random_state)
}


class CustomAnalysisService(metaclass=SingletonMeta):
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

        df[target_column] = df[target_column].replace(
            {
                'Positivo': '1',
                'Negativo': '-1',
                'Neutral': '0'
            }
        )

        return df

    def calculate_accuracy(self, y_test, y_pred) -> tuple:
        confusion = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print('Accuracy calculado.')
        return confusion, accuracy, report

    def predict(self, X_test, model):
        y_pred = model.predict(X_test)
        print('Predicción finalizada.')
        return y_pred

    def execute_algorithm(self, classifier_name: str, X_train, X_test, y_train, y_test):
        print(f'\n-------------------\n{classifier_name}\n-------------------')
        classifiers[classifier_name].fit(X_train, y_train)
        print(f'Modelo entrenado.')
        y_pred = self.predict(X_test, classifiers[classifier_name])
        confusion, accuracy, report = self.calculate_accuracy(y_test, y_pred)
        print(f'Confusion matrix:\n {confusion}')
        print(f'Accuracy score: {accuracy}')
        print(f'Classification report:\n {report}')
        # fig, ax = plt.subplots(figsize=(10, 10))

        # plot_confusion_matrix(
        #     classifiers[classifier_name],
        #     X_test,
        #     y_test,
        #     ax=ax,
        #     cmap='Blues'
        # )
        print()

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

        for classifier in classifiers:
            self.execute_algorithm(
                classifier, self.X_train, self.X_test, self.y_train, self.y_test)

        # for index, row in df.iterrows():
        #     print(f'classifying item {index + 1} of {len(df)}')
        #     text = row[text_column]
        #     sentiment = row[target_column]

        #     scores = [
        #         classifiers['nearest_neighbors'].classify(text),
        #         classifiers['linear_svm'].classify(text),
        #         classifiers['rbf_svm'].classify(text),
        #         classifiers['gaussian_process'].classify(text),
        #         classifiers['decision_tree'].classify(text),
        #         classifiers['random_forest'].classify(text),
        #         classifiers['neural_net'].classify(text),
        #         classifiers['ada_boost'].classify(text),
        #         classifiers['naive_bayes'].classify(text),
        #         classifiers['qda'].classify(text),
        #         classifiers['gradient_boosting'].classify(text),
        #         classifiers['logistic_regression'].classify(text),
        #         classifiers['stochastic_gradient_descent'].classify(text),
        #     ]

        #     data.append([
        #         text,
        #         sentiment,
        #         scores[0],
        #         scores[1],
        #         scores[2],
        #         scores[3],
        #         scores[4],
        #         scores[5],
        #         scores[6],
        #         scores[7],
        #         scores[8],
        #         scores[9],
        #         scores[10],
        #         scores[11],
        #         scores[12],
        #         self.dataset_service.most_frequent(scores)
        #     ])

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
                'stochastic_gradient_descent',
                'max_voting'
            ]
        )
