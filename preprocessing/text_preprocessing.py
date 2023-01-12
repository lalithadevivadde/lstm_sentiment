import numpy as np
import pandas as pd
import string
from unidecode import unidecode

class TextPreprocessor:
    def __init__(self, remove_punct: bool = True, remove_digits: bool = True,
                 remove_stop_words: bool = True,
                 remove_short_words: bool = True, minlen: int = 1, maxlen: int = 1, top_p: float = None,
                 bottom_p: float = None):
        self.remove_punct = remove_punct
        self.remove_digits = remove_digits
        self.remove_stop_words = remove_stop_words
        self.remove_short_words = remove_short_words
        self.minlen = minlen
        self.maxlen = maxlen
        self.top_p = top_p
        self.bottom_p = bottom_p
        self.words_to_remove = []
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                           'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                           'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                           'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                           'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                           'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'if', 'or',
                           'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                           'into', 'through', 'during', 'before', 'after', 'to', 'from',
                           'in', 'out', 'on', 'off', 'further', 'then', 'once',
                           'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                           'other', 'such', 'only', 'own', 'same', 'so', 'than',
                           'too', 'can', 'will', 'just', 'should',
                           'now']

        self.contraction_to_expansion = {"ain't": "am not",
                                         "aren't": "are not",
                                         "can't": "cannot",
                                         "can't've": "cannot have",
                                         "'cause": "because",
                                         "could've": "could have",
                                         "couldn't": "could not",
                                         "couldn't've": "could not have",
                                         "didn't": "did not",
                                         "doesn't": "does not",
                                         "don't": "do not",
                                         "hadn't": "had not",
                                         "hadn't've": "had not have",
                                         "hasn't": "has not",
                                         "haven't": "have not",
                                         "he'd": "he would",
                                         "he'd've": "he would have",
                                         "he'll": "he will",
                                         "he'll've": "he will have",
                                         "he's": "he is",
                                         "how'd": "how did",
                                         "how'd'y": "how do you",
                                         "how'll": "how will",
                                         "how's": "how is",
                                         "i'd": "i would",
                                         "i'd've": "i would have",
                                         "i'll": "i will",
                                         "i'll've": "i will have",
                                         "i'm": "i am",
                                         "i've": "i have",
                                         "isn't": "is not",
                                         "it'd": "it had",
                                         "it'd've": "it would have",
                                         "it'll": "it will",
                                         "it'll've": "it will have",
                                         "it's": "it is",
                                         "let's": "let us",
                                         "ma'am": "madam",
                                         "mayn't": "may not",
                                         "might've": "might have",
                                         "mightn't": "might not",
                                         "mightn't've": "might not have",
                                         "must've": "must have",
                                         "mustn't": "must not",
                                         "mustn't've": "must not have",
                                         "needn't": "need not",
                                         "needn't've": "need not have",
                                         "o'clock": "of the clock",
                                         "oughtn't": "ought not",
                                         "oughtn't've": "ought not have",
                                         "shan't": "shall not",
                                         "sha'n't": "shall not",
                                         "shan't've": "shall not have",
                                         "she'd": "she would",
                                         "she'd've": "she would have",
                                         "she'll": "she will",
                                         "she'll've": "she will have",
                                         "she's": "she is",
                                         "should've": "should have",
                                         "shouldn't": "should not",
                                         "shouldn't've": "should not have",
                                         "so've": "so have",
                                         "so's": "so is",
                                         "that'd": "that would",
                                         "that'd've": "that would have",
                                         "that's": "that is",
                                         "there'd": "there had",
                                         "there'd've": "there would have",
                                         "there's": "there is",
                                         "they'd": "they would",
                                         "they'd've": "they would have",
                                         "they'll": "they will",
                                         "they'll've": "they will have",
                                         "they're": "they are",
                                         "they've": "they have",
                                         "to've": "to have",
                                         "wasn't": "was not",
                                         "we'd": "we had",
                                         "we'd've": "we would have",
                                         "we'll": "we will",
                                         "we'll've": "we will have",
                                         "we're": "we are",
                                         "we've": "we have",
                                         "weren't": "were not",
                                         "what'll": "what will",
                                         "what'll've": "what will have",
                                         "what're": "what are",
                                         "what's": "what is",
                                         "what've": "what have",
                                         "when's": "when is",
                                         "when've": "when have",
                                         "where'd": "where did",
                                         "where's": "where is",
                                         "where've": "where have",
                                         "who'll": "who will",
                                         "who'll've": "who will have",
                                         "who's": "who is",
                                         "who've": "who have",
                                         "why's": "why is",
                                         "why've": "why have",
                                         "will've": "will have",
                                         "won't": "will not",
                                         "won't've": "will not have",
                                         "would've": "would have",
                                         "wouldn't": "would not",
                                         "wouldn't've": "would not have",
                                         "y'all": "you all",
                                         "y'alls": "you alls",
                                         "y'all'd": "you all would",
                                         "y'all'd've": "you all would have",
                                         "y'all're": "you all are",
                                         "y'all've": "you all have",
                                         "you'd": "you had",
                                         "you'd've": "you would have",
                                         "you'll": "you you will",
                                         "you'll've": "you you will have",
                                         "you're": "you are",
                                         "you've": "you have"
                                         }

    @staticmethod
    def __remove_double_whitespaces(string: str):
        return " ".join(string.split())

    def __remove_url(self, string_series: pd.Series):
        clean_string_series = string_series.str.replace(
            pat=r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
            repl=" ", regex=True).copy()
        return clean_string_series.map(self.__remove_double_whitespaces)

    def __expand(self, string_series: pd.Series):
        clean_string_series = string_series.copy()
        for c, e in self.contraction_to_expansion.items():
            clean_string_series = clean_string_series.str.replace(pat=c, repl=e, regex=False).copy()
        return clean_string_series.map(self.__remove_double_whitespaces)

    def __remove_punct(self, string_series: pd.Series):
        clean_string_series = string_series.copy()
        puncts = [r'\n', r'\r', r'\t']
        puncts.extend(list(string.punctuation))
        for i in puncts:
            clean_string_series = clean_string_series.str.replace(pat=i, repl=" ", regex=False).copy()
        return clean_string_series.map(self.__remove_double_whitespaces)

    def __remove_digits(self, string_series: pd.Series):
        clean_string_series = string_series.str.replace(pat=r'\d', repl=" ", regex=True).copy()
        return clean_string_series.map(self.__remove_double_whitespaces)

    @staticmethod
    def __remove_short_words(string_series: pd.Series, minlen: int = 1, maxlen: int = 1):
        clean_string_series = string_series.map(lambda string: " ".join([word for word in string.split() if
                                                                         (len(word) > maxlen) or (len(word) < minlen)]))
        return clean_string_series

    def __remove_stop_words(self, string_series: pd.Series):
        def str_remove_stop_words(string: str):
            stops = self.stop_words
            return " ".join([token for token in string.split() if token not in stops])

        return string_series.map(str_remove_stop_words)

    def __remove_top_bottom_words(self, string_series: pd.Series, top_p: int = None,
                                  bottom_p: int = None, dataset: str = 'train'):
        if dataset == 'train':
            if top_p is None:
                top_p = 0
            if bottom_p is None:
                bottom_p = 0

            if top_p > 0 or bottom_p > 0:
                word_freq = pd.Series(" ".join(string_series).split()).value_counts()
                n_words = len(word_freq)

            if top_p > 0:
                self.words_to_remove.extend([*word_freq.index[: int(np.ceil(top_p * n_words))]])

            if bottom_p > 0:
                self.words_to_remove.extend([*word_freq.index[-int(np.ceil(bottom_p * n_words)):]])

        if len(self.words_to_remove) == 0:
            return string_series
        else:
            clean_string_series = string_series.map(lambda string: " ".join([word for word in string.split()
                                                                             if word not in self.words_to_remove]))
            return clean_string_series

    def preprocess(self, string_series: pd.Series, dataset: str = "train"):
        string_series = string_series.str.lower().copy()
        string_series = string_series.map(unidecode).copy()
        string_series = self.__remove_url(string_series=string_series)
        string_series = self.__expand(string_series=string_series)

        if self.remove_punct:
            string_series = self.__remove_punct(string_series=string_series)
        if self.remove_digits:
            string_series = self.__remove_digits(string_series=string_series)
        if self.remove_stop_words:
            string_series = self.__remove_stop_words(string_series=string_series)
        if self.remove_short_words:
            string_series = self.__remove_short_words(string_series=string_series,
                                                      minlen=self.minlen,
                                                      maxlen=self.maxlen)
        string_series = self.__remove_top_bottom_words(string_series=string_series,
                                                       top_p=self.top_p,
                                                       bottom_p=self.bottom_p, dataset=dataset)

        string_series = string_series.str.strip().copy()
        string_series.replace(to_replace="", value="this is an empty message", inplace=True)

        return string_series