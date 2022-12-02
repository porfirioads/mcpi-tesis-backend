from typing import List
from app.utils.singleton import SingletonMeta
from app.utils.datasets import read_dataset
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer


class AnalysisService(metaclass=SingletonMeta):
    sia = SentimentIntensityAnalyzer()

    def prepare_dataset(self, file_path: str) -> pd.DataFrame:
        df = read_dataset(file_path=file_path, encoding='utf-8', delimiter='|')
        df['sentiment'] = df['sentiment'].replace(
            {
                'Positivo': 'pos',
                'Negativo': 'neg',
                'Neutral': 'neu'
            }
        )
        return df

    def vader(self, file_path: str):
        df = self.prepare_dataset(file_path)

    def get_vader_score(self, text: str) -> int:
        score = self.sia.polarity_scores(text)['compound']
        return score

    def get_textblob_score(self, text: str) -> int:
        return TextBlob(text).sentiment.polarity

    def get_textblob_naive_bayes_score(self, text: str):
        sentiment = TextBlob(text, analyzer=NaiveBayesAnalyzer()).sentiment
        pos_score = sentiment.p_pos
        neg_score = sentiment.p_neg
        score_dif = pos_score - neg_score

        if (abs(pos_score - neg_score) <= 0.15):
            return score_dif

        if (pos_score > neg_score):
            return pos_score

        return - neg_score

    def get_classified_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        classifiers: List[str]
    ) -> pd.DataFrame:
        pass
