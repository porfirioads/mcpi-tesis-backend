import pandas as pd


def read_dataset(file_path: str, encoding: str, delimiter: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
    return df
