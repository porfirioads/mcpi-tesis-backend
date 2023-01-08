import os
from app.schemas.common_schemas import FileUpload
from app.utils.singleton import SingletonMeta
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, \
    precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from app.config import logger


class DatasetService(metaclass=SingletonMeta):
    def read_dataset(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        logger.debug('DatasetService.read_dataset()')
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        return df

    def split_dataset(self, df: pd.DataFrame, y_true: str) -> tuple:
        logger.debug('DatasetService.split_dataset()')

        X_train, X_test, y_train, y_test = train_test_split(
            # df.drop[y_true],
            df,
            df[y_true],
            test_size=0.25,
            random_state=1
        )

        return (X_train, X_test, y_train, y_test)

    def to_csv(self, df: pd.DataFrame, file_path: str) -> FileUpload:
        logger.debug('DatasetService.to_csv()')

        df.to_csv(
            file_path,
            index=False
        )

        file_path = file_path.split('/')[-1]

        return FileUpload(file_path=file_path)

    def summary_dataset(
        self,
        file_path: str,
        encoding: str,
        delimiter,
        target_column: str
    ) -> dict:
        logger.debug('DatasetService.summary_dataset()')
        os.stat(f'resources/cleaned/{file_path}')

        df = self.read_dataset(
            file_path=f'resources/cleaned/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )

        return df[target_column].value_counts().to_dict()

    def get_metrics(
        self,
        df: pd.DataFrame,
        y_true: str,
        y_pred: str,
        has_probability: bool = False
    ) -> dict:
        logger.debug('DatasetService.get_metrics()')
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index, row in df.iterrows():
            if row[y_true] == 1:
                if row[y_pred] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if row[y_pred] == -1:
                    tn += 1
                else:
                    fn += 1

        accuracy = accuracy_score(df[y_true], df[y_pred])

        kappa = cohen_kappa_score(df[y_true], df[y_pred])

        f1 = f1_score(
            df[y_true],
            df[y_pred],
            pos_label=1,
            average='binary'
        )

        precision = precision_score(
            df[y_true],
            df[y_pred],
            pos_label=1,
            average='binary'
        )

        sensitivity = recall_score(
            df[y_true],
            df[y_pred],
            pos_label=1,
            average='binary'
        )

        specificity = recall_score(
            df[y_true],
            df[y_pred],
            pos_label=-1,
            average='binary'
        )

        metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'kappa': kappa,
            'f1': f1,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

        if has_probability:
            items = []

            for item in df[f'{y_pred}_proba'].items():
                proba = item[1].replace('[', '').replace(']', '')
                items.append(np.fromstring(proba, dtype=float, sep=' '))

            y_proba = np.array(items)

            if np.isnan(y_proba).any():
                metrics['auc'] = np.nan
            else:
                metrics['auc'] = roc_auc_score(
                    df[y_true],
                    y_proba[:, 1],
                    average=None
                )

        return metrics

    def metrics_to_df(self, metrics: dict) -> pd.DataFrame:
        logger.debug('DatasetService.metrics_to_df()')
        data = []
        headers = ['algorithm']

        for algorithm in metrics:
            for metric in metrics[algorithm]:
                headers.append(metric)
            break

        for algorithm in metrics:
            data.append([algorithm])
            for metric in metrics[algorithm]:
                data[-1].append(metrics[algorithm][metric])

        df = pd.DataFrame(
            data,
            columns=headers
        )

        return df

    def get_dataset_summary(self, df: pd.DataFrame) -> dict:
        return {
            'row_count': df.shape[0],
            'col_count': df.shape[1],
        }
