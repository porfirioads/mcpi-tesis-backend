import pandas as pd
from sklearn.model_selection import train_test_split
from app.patterns.singleton import SingletonMeta
from app.config import logger
from app.schemas.file_schemas import FileUpload


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
