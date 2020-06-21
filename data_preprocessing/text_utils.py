from difflib import SequenceMatcher

from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams


class TextUtils(object):
    _tokenizer = RegexpTokenizer(r"\w+")

    @staticmethod
    def string_to_list(string, split_char):
        return string.split(split_char)

    @staticmethod
    def create_ngrams(text):
        n_grams = list()
        for i in range(1, len(text) + 1):
            n_gram = ngrams(text, i)

            for gram in n_gram:
                n_grams.append(" ".join(gram))

        return n_grams

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def preprocess(string):
        tokens = TextUtils._tokenizer.tokenize(string)
        tokens = [t.lower() for t in tokens]
        return tokens

    @staticmethod
    def find_max_length(list_):

        max_len = -1

        for i in list_:
            if max_len < len(i):
                max_len = len(i)
                max_context = i

        return max_len, max_context
