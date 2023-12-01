import re
from typing import Dict, Set, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


class SentenceSplitter:
    def __init__(self):
        self.abbreviation_set = self.load_abbreviations()
        self.web_words = ("http:", ".html", "www", ".tr", ".edu", ".net", ".gov", "@", ".com", ".org")
        self.abbreviations = self.abbreviation_set.union(self.web_words)
        self.punctuation = set(".?!")
        self.model = LogisticRegression()

    @staticmethod
    def load_abbreviations() -> Set[str]:
        lower_map = {
            ord(u'I'): u'ı',
            ord(u'İ'): u'i',
        }

        abbreviation_set = set()
        with open('../data/abbreviation_list.txt', 'r', encoding="utf-8") as f:
            lines = list(f.readlines())
            for line in lines:
                if len(line.strip()) > 0:
                    abbr = re.sub(r'\s+', "", line.strip())
                    abbreviation_set.add(re.sub(r'\.$', "", abbr))
                    abbr = abbr.translate(lower_map)
                    abbreviation_set.add(re.sub(r'\.$', "", abbr.lower()))

        return abbreviation_set

    def create_features(self, text):
        total_number_of_spaces = len(text.split()) - 1
        location_of_spaces = (pd.Series(text.split()).str.len().cumsum().values + np.arange(0,
                                                                                            len(text.split())))[:-1]

        features = []
        for loc in location_of_spaces:
            is_prev_char_sent_ender_punc = text[loc - 1] in self.punctuation
            is_next_char_capital_letter = text[loc + 1].isupper()
            is_next_char_numeric = text[loc + 1].isnumeric()
            is_prev_char_numeric = text[loc - 1].isnumeric()
            is_prev_token_abbreviation = text[:loc].split()[-1] in self.abbreviations

            features.append([is_prev_char_sent_ender_punc, is_next_char_capital_letter, is_next_char_numeric,
                             is_prev_char_numeric, is_prev_token_abbreviation])
        return np.array(features) * 1, location_of_spaces

    @staticmethod
    def create_labels(text_list):
        total_number_of_spaces = len(" ".join(text_list).split()) - 1
        y = np.zeros(total_number_of_spaces)
        split_positions = (pd.Series(text_list).str.split().str.len().cumsum() - 1).values[:-1]
        y[split_positions] = 1
        return y

    def fit(self, text):
        X, _ = self.create_features(text)
        y = self.create_labels(text.split())
        self.model.fit(X, y)

    def predict(self, text):
        X, location_of_spaces = self.create_features(text)
        preds = self.model.predict(X)
        return [int(i) for i in preds]

    def rule_based_split(self, text):
        abbrev_pattern = '|'.join(re.escape(abbr) for abbr in self.abbreviations) + r'|\d+'
        pattern = rf"(?<!\b(?:{abbrev_pattern}))(?<!\.\.)[.?!]\s+"
        sentences = re.split(pattern, text, flags=re.UNICODE)
        return sentences

    def ml_based_split(self, text):
        X, location_of_spaces = self.create_features(text)
        preds = self.model.predict(X)
        boolean_preds = [x == 1 for x in preds]
        indices = [0] + location_of_spaces[boolean_preds].tolist()
        sentences = [text[i:j] for i, j in zip(indices, indices[1:] + [None])]
        sentences = [sentence if sentence[0] != ' ' else sentence[1:] for sentence in sentences]
        return sentences
