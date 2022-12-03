from typing import List, Callable
from app.utils.singleton import SingletonMeta
from app.utils.datasets import read_dataset
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from collections import Counter


class AnalysisService(metaclass=SingletonMeta):
    sia = SentimentIntensityAnalyzer()

    def prepare_dataset(self, file_path: str) -> pd.DataFrame:
        df = read_dataset(
            file_path=f'uploads/cleaned/{file_path}',
            encoding='utf-8',
            delimiter='|'
        )

        df['sentiment'] = df['sentiment'].replace(
            {
                'Positivo': '1',
                'Negativo': '-1',
                'Neutral': '0'
            }
        )

        return df

    def most_frequent(self, items):
        occurence_count = Counter(items)
        return occurence_count.most_common(1)[0][0]

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

    def classify_pretrained(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> pd.DataFrame:
        data = []

        for i in range(len(df)):
            print(f'classifying item {i + 1} of {len(df)}')
            text = df.loc[i, text_column]
            sentiment = df.loc[i, target_column]

            scores = [
                self.get_vader_score(str(text)),
                self.get_textblob_score(str(text)),
                self.get_textblob_naive_bayes_score(str(text)),
            ]

            for i in range(len(scores)):
                if scores[i] > 0.15:
                    scores[i] = 1
                elif scores[i] < -0.15:
                    scores[i] = -1
                else:
                    scores[i] = 0

            data.append([
                text,
                sentiment,
                scores[0],
                scores[1],
                scores[2],
                self.most_frequent(scores)
            ])

        return pd.DataFrame(
            data,
            columns=[
                'answer',
                'sentiment',
                'vader',
                'textblob',
                'naive_bayes',
                'max_voting'
            ]
        )
