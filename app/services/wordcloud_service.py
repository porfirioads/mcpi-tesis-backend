from app.patterns.singleton import SingletonMeta
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class WordcloudService(metaclass=SingletonMeta):
    def generate_wordcloud(
        self,
        text: str,
        title: str,
        file_path: str
    ):
        # Configure the wordcloud
        wordcloud = WordCloud(width=1500, height=500,
                              background_color='white',
                              min_font_size=10).generate(text)

        # Save the wordcloud
        plt.figure(figsize=(15, 5), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title(
            title,
            fontdict={
                'family': 'sans',
                'color': 'black',
                'size': 50
            },
            pad=20
        )
        plt.savefig(file_path)
        plt.close()
