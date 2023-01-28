import pandas as pd
from app.patterns.singleton import SingletonMeta


class AnalysisService(metaclass=SingletonMeta):
    def logistic_regression(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        pass

    def naive_bayes(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        pass

    def svm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ) -> pd.DataFrame:
        pass
