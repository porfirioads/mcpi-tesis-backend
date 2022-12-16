import os
from typing import List
from fastapi import UploadFile, HTTPException
from app.schemas.common_schemas import FileUpload
from app.utils.singleton import SingletonMeta
from app.utils.strings import Strings
from fastapi.responses import FileResponse
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, \
    precision_score, recall_score
from collections import Counter
from sklearn.model_selection import train_test_split
from app.config import logger


class DatasetService(metaclass=SingletonMeta):
    def most_frequent(self, items: List):
        occurence_count = Counter(items)
        return occurence_count.most_common(1)[0][0]

    def read_dataset(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        return df

    def split_dataset(self, df: pd.DataFrame, y_true: str) -> tuple:
        y = df[y_true]

        X_train, X_test, y_train, y_test = train_test_split(
            df,
            y,
            test_size=0.25,
            random_state=1
        )

        return (X_train, X_test, y_train, y_test)

    def to_csv(self, df: pd.DataFrame, file_path: str) -> FileUpload:
        timestamp = int(datetime.timestamp(datetime.now()))
        file_path = f'{file_path[0: -4]}_{timestamp}.csv'

        df.to_csv(
            file_path,
            index=False
        )

        file_path = file_path.split('/')[-1]

        return FileUpload(file_path=file_path)

    def upload_dataset(self, file: UploadFile) -> FileUpload:
        if file.content_type != 'text/csv':
            raise HTTPException(
                status_code=400, detail='DATASET_MUST_BE_A_CSV')

        timestamp = int(datetime.timestamp(datetime.now()))
        snack_name = Strings.to_snack_case(file.filename)
        file_path = f'{snack_name[0: -4]}_{timestamp}.csv'

        with open(f'resources/uploads/{file_path}', 'wb') as f:
            f.write(file.file.read())

        return FileUpload(file_path=file_path)

    def get_datasets_list(self, path: str) -> List[FileUpload]:
        files = os.listdir(f'resources/{path}')

        if '.gitkeep' in files:
            files.remove('.gitkeep')

        if 'cleaned' in files:
            files.remove('cleaned')

        if 'classified' in files:
            files.remove('classified')

        return [
            FileUpload(file_path=f'{file_name}') for file_name in files
        ]

    def download_dataset(self, file_path: str) -> FileResponse:
        os.stat(f'resources/{file_path}')
        return FileResponse(
            path=f'resources/{file_path}',
            media_type='text/csv',
            filename=file_path
        )

    def summary_dataset(
        self,
        file_path: str,
        encoding: str,
        delimiter,
        target_column: str
    ) -> dict:
        os.stat(f'resources/cleaned/{file_path}')
        df = self.read_dataset(
            file_path=f'resources/cleaned/{file_path}',
            encoding=encoding,
            delimiter=delimiter
        )
        return df[target_column].value_counts().to_dict()

    def get_metrics(self, df: pd.DataFrame, y_true: str, y_pred: str) -> dict:
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for index, row in df.iterrows():
            logger.debug(f'{ row[y_true]} vs { row[y_pred]}')
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

        logger.debug(f'tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')

        accuracy = accuracy_score(df[y_true], df[y_pred])

        kappa = cohen_kappa_score(df[y_true], df[y_pred])

        f1 = f1_score(
            df[y_true],
            df[y_pred],
            pos_label=1,
            average='micro'
        )

        precision = precision_score(
            df[y_true],
            df[y_pred],
            pos_label=1,
            average='micro'
        )

        recall = recall_score(
            df[y_true],
            df[y_pred],
            pos_label=1,
            average='micro'
        )

        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'f1': f1,  # Promedio de la precisión
            'precision': precision,  # Cuantos son relevantes
            'recall': recall,  # Cuantos son seleccionados
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
            # 'sensitivity': (tp) / (tp + fn),
            # 'specificity': 1 - ((fp) / (fp + tn)),
            # 'accuracy_calc': (tp + tn) / (tp + fp + fn + tn)
        }
