import re
import string
from random import randint
from app.patterns.singleton import SingletonMeta
import nltk.data
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class NltkService(metaclass=SingletonMeta):
    def download(self):
        nltk.download([
            'names',
            'stopwords',
            'state_union',
            'twitter_samples',
            'movie_reviews',
            'averaged_perceptron_tagger',
            'vader_lexicon',
            'punkt',
            'wordnet',
            'omw-1.4'
        ])

    def extract_synonyms(self, text: str):
        output = ''

        # Get the list of words from the entire text
        words = word_tokenize(text)

        # Identify the parts of speech
        tagged = nltk.pos_tag(words)

        for i in range(0, len(words)):
            replacements = []

            # Only replace nouns with nouns, vowels with vowels etc.
            for syn in wordnet.synsets(words[i]):

                # Do not attempt to replace proper nouns or determiners
                if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
                    break

                # The tokenizer returns strings like NNP, VBP etc
                # but the wordnet synonyms has tags like .n.
                # So we extract the first character from NNP ie n
                # then we check if the dictionary word has a .n. or not
                word_type = tagged[i][1][0].lower()
                if syn.name().find("." + word_type + "."):
                    # extract the word only
                    r = syn.name()[0:syn.name().find(".")]
                    replacements.append(r)

            if len(replacements) > 0:
                # Choose a random replacement
                replacement = replacements[randint(0, len(replacements) - 1)]
                output = output + " " + replacement
            else:
                # If no replacement could be found, then just use the
                # original word
                output = output + " " + words[i]

        return output

    def remove_stop_words(self, text: str) -> str:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_words = [w for w in word_tokens if not w.lower() in stop_words]
        return ' '.join(filtered_words)

    def remove_punctuation(self, text: str) -> str:
        puntuaction = string.punctuation
        word_tokens = word_tokenize(text)
        filtered_words = [
            w for w in word_tokens if not w.lower() in puntuaction]
        return ' '.join(filtered_words)

    def remove_numbers(self, text: str) -> str:
        return re.sub(r'\d+', '', text)

    def stem_words(self, text: str) -> str:
        stemmer = SnowballStemmer(language='english')
        word_tokens = word_tokenize(text)
        stemmed_words = [stemmer.stem(w) for w in word_tokens]
        return ' '.join(stemmed_words)

    def to_lower(self, text: str) -> str:
        return text.lower()
