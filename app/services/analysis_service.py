from app.utils.singleton import SingletonMeta
from app.utils.datasets import read_dataset


class AnalysisService(metaclass=SingletonMeta):
    def vader(file_path: str):
        df = read_dataset(file_path=file_path, encoding='utf-8', delimiter='|')
        df['sentiment'] = df['sentiment'].replace(
            {
                'Positivo': 'pos',
                'Negativo': 'neg',
                'Neutral': 'neu'
            }
        )
