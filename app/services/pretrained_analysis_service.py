
from app.services.dataset_service import DatasetService
from app.utils.singleton import SingletonMeta
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.classifiers import MaxEntClassifier, DecisionTreeClassifier, NaiveBayesClassifier, PositiveNaiveBayesClassifier
from textblob.sentiments import NaiveBayesAnalyzer
from typing import List
import pandas as pd


class PretrainedAnalysisService(metaclass=SingletonMeta):
    sia = SentimentIntensityAnalyzer()
    dataset_service = DatasetService()

    def most_frequent(self, items: List):
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

        for index, row in df.iterrows():
            print(f'classifying item {index + 1} of {len(df)}')
            text = row[text_column]
            sentiment = row[target_column]

            scores = [
                self.get_vader_score(str(text)),
                self.get_textblob_score(str(text)),
                self.get_textblob_naive_bayes_score(str(text)),
            ]

            for i in range(len(scores)):
                if scores[i] > 0.3:
                    scores[i] = 1
                elif scores[i] < -0.3:
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