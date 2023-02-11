import pandas as pd
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

    def to_csv(self, df: pd.DataFrame, file_path: str) -> FileUpload:
        logger.debug('DatasetService.to_csv()')

        df.to_csv(
            file_path,
            index=False
        )

        file_path = file_path.split('/')[-1]

        return FileUpload(file_path=file_path)

    def get_dataset_summary(self, df: pd.DataFrame) -> dict:
        return {
            'row_count': df.shape[0],
            'col_count': df.shape[1],
        }
