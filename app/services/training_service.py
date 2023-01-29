from typing import Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from app.patterns.singleton import SingletonMeta
from app.services.dataset_service import DatasetService
from app.services.metrics_service import MetricsService

dataset_service = DatasetService()
metrics_service = MetricsService()


class TrainingService(metaclass=SingletonMeta):
    def execute_algorithm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str,
        model: Any,
        plot_title: str,
        model_suffix: str
    ) -> dict:
        # Read dataset
        df = dataset_service.read_dataset(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        # Get word column names
        word_columns = df.columns[2:].to_list()

        # Convert word columns to categories
        for column in word_columns:
            df[column] = df[column].astype('category').cat.codes

        # Convert sentiment column to categories
        df['sentiment'] = df['sentiment'].replace(
            {'Positivo': 1, 'Negativo': -1}
        )

        # Split dataset in training and test
        x_values = df[word_columns]
        y_values = df['sentiment']
        seed = 1

        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=0.3,
            random_state=seed
        )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)
        y_proba = [str(proba) for proba in y_proba]
        file_path = f'{file_path.split("/")[-1][:-4]}.png'
        auroc_file_path = f'auroc_{model_suffix}_{file_path}'
        confusion_file_path = f'confusion_{model_suffix}_{file_path}'

        metrics_service.plot_aucroc(
            model=model,
            x_test=x_test,
            y_test=y_test,
            title=f'{plot_title} auroc',
            file_path=f'resources/metrics/{auroc_file_path}'
        )

        metrics = metrics_service.get_metrics(
            y_test=y_test,
            y_pred=y_pred
        )

        metrics_service.plot_confusion_matrix(
            y_test=y_test,
            y_pred=y_pred,
            title=f'{plot_title} confusion matrix',
            file_path=f'resources/metrics/{confusion_file_path}'
        )

        columns = ['answer', 'sentiment', model_suffix, f'{model_suffix}_proba']
        new_df = pd.DataFrame(columns=columns)
        new_df['answer'] = df.iloc[y_test.index.values, :]['answer'].values
        new_df['sentiment'] = y_test.values
        new_df[model_suffix] = y_pred
        new_df[f'{model_suffix}_proba'] = y_proba

        return {
            'df': new_df,
            'auroc_file_path': auroc_file_path,
            'confusion_file_path': confusion_file_path,
            'metrics': metrics
        }

    def logistic_regression(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        result = self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model=LogisticRegression(solver='liblinear', random_state=0),
            plot_title='Logistic Regression L1-L2',
            model_suffix='lgr'
        )

        return result['df']

    def naive_bayes(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        result = self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model=BernoulliNB(),
            plot_title='Naive-Bayes Bernoulli',
            model_suffix='nbb'
        )

        return result['df']

    def svm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        result = self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model=SVC(kernel='linear', degree=3, probability=True),
            plot_title='Linear SVM',
            model_suffix='svm'
        )

        return result['df']

    def assemble(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        # Do analysis with naive bayes
        df_nbb = self.naive_bayes(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        # Do analysis with logistic regression
        df_lgr = self.logistic_regression(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        # Do analysis with support vector machine
        df_svm = self.svm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        columns = [
            'answer',
            'sentiment',
            'nbb',
            'nbb_proba',
            'lgr',
            'lgr_proba',
            'svm',
            'svm_proba',
            'max'
        ]

        # Generate new dataframe
        new_df = pd.DataFrame(columns=columns)
        new_df['answer'] = df_nbb['answer']
        new_df['sentiment'] = df_nbb['sentiment']
        new_df['nbb'] = df_nbb['nbb']
        new_df['nbb_proba'] = df_nbb['nbb_proba']
        new_df['lgr'] = df_lgr['lgr']
        new_df['lgr_proba'] = df_lgr['lgr_proba']
        new_df['svm'] = df_svm['svm']
        new_df['svm_proba'] = df_svm['svm_proba']
        new_df['max'] = new_df[['nbb', 'lgr', 'svm']].mode(axis=1).iloc[:, 0]

        return new_df
