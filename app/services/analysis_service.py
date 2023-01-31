import joblib
from app.patterns.singleton import SingletonMeta
from app.services.dataset_service import DatasetService
from app.services.nltk_service import NltkService
import matplotlib.pyplot as plt

dataset_service = DatasetService()
nltk_service = NltkService()


class AnalysisService(metaclass=SingletonMeta):
    def execute_algorithm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str,
        model_file_path: str
    ):
        df = dataset_service.read_dataset(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
        )

        questions = [
            'Do you have other climate mitigation ideas? Submit here:',
            'Do you have other climate adaptation ideas? Submit here:',
            'Do you have other ideas for environmental equity, justice,'
            + ' and community resilience? Submit here:',
            'Do you have other policy ideas? Submit here:',
            'Are you interested in participating in any other ways to help'
            + ' make Tucson environmentally sustainable? Submit here:',
            'Is there anything else you would like to share that was not'
            + ' already addressed?'
        ]

        model = joblib.load(model_file_path)

        print(model)

        for question in questions:
            opinions = df[question].str.strip().dropna(how='all')\
                .apply(nltk_service.to_lower)\
                .apply(nltk_service.remove_stop_words)\
                .apply(nltk_service.remove_punctuation)\
                .apply(nltk_service.remove_numbers)\
                .apply(nltk_service.stem_words)

            for opinion in opinions:
                print(opinion)
                # model.predict(opinion)

    def generate_wordcloud(
        self,
        text: str,
        title: str,
        file_path: str
    ):
        from wordcloud import WordCloud

        # Configure the wordcloud
        wordcloud = WordCloud(width=1500, height=500,
                              background_color='white',
                              min_font_size=10).generate(text)

        # Save the wordcloud
        plt.figure(figsize=(15, 5), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.title(title, fontdict={
            'family': 'sans', 'color': 'black', 'size': 50}, pad=20)

        plt.savefig(file_path)
        plt.close()

    def logistic_regression(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        return self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model_file_path='resources/models/lgr.sav'
        )

    def naive_bayes(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        return self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model_file_path='resources/models/nbb.sav'
        )

    def svm(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        return self.execute_algorithm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter,
            model_file_path='resources/models/svm.sav'
        )

    def assemble(
        self,
        file_path: str,
        encoding: str,
        delimiter: str
    ):
        self.logistic_regression(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        self.naive_bayes(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )

        self.svm(
            file_path=file_path,
            encoding=encoding,
            delimiter=delimiter
        )
