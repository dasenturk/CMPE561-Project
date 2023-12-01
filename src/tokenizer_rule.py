import re
import nltk
from typing import Dict, Set, List
from sentence_splitter import SentenceSplitter


class RuleBasedTokenizer:
    def __init__(self):
        self.abbreviation_set = self.load_abbreviations()
        self.mwe_lexicon = self.load_mwe_lexicon()

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

    @staticmethod
    def load_mwe_lexicon() -> Set[str]:
        mwe_lexicon = set()
        with open("../data/MWE_Lexicon.txt", "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if (i == len(lines) - 1):
                    mwe_lexicon.add(lines[i])
                else:
                    mwe_lexicon.add(lines[i][:len(lines[i]) - 1])
        return mwe_lexicon

    @staticmethod
    def preprocess(text):
        text = re.sub('-(\n+)', '', text)
        text = re.sub('(\s+)', ' ', text)
        return text

    def find_split_points(self, text):
        split_points = set()
        punctuations = set(
            ["?", "!", "…", ":", ",", ";", ".", '"', "'", "\\", "/", "(", ")", "[", "]", "‘", "’", "“", "”"])
        white_space_places = []
        immunes = [0 for i in range(len(text))]
        email_matches = re.finditer('[-\w\.]+@([\w-]+\.)+[\w-]+', text)
        email_spans = [match.span() for match in email_matches]
        # url regexini değiştirmek gerekebilir
        url_matches = re.finditer(
            '(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', text)
        # url regex is from here: https://stackoverflow.com/questions/30970068/js-regex-url-validation
        url_spans = [match.span() for match in url_matches]
        currency_matches = re.finditer('(-)?([0-9]{1,3}([\.,]+[0-9]{3})*([\.,]+[0-9]+)?|[\.,]+[0-9]+)\s[₺\$\u20AC£]{1}',
                                       text)
        currency_spans = [match.span() for match in currency_matches]
        currency_matches_2 = re.finditer('[₺\$\u20AC£]{1}(-)?([0-9]{1,3}([\.,]+[0-9]{3})*([\.,]+[0-9]+)?|[\.,]+[0-9]+)',
                                         text)
        currency_spans_2 = [match.span() for match in currency_matches_2]
        number_matches = re.finditer('(-)?([0-9]{1,3}([\.,]+[0-9]{3})*([\.,]+[0-9]+)?|[\.,]+[0-9]+)', text)
        number_spans = [match.span() for match in number_matches]
        date_matches = re.finditer('(0[1-9]|[12][0-9]|3[01])[-\/.](0[1-9]|1[1,2])[-\/.]\d+', text)
        # date regex is a modified version of https://www.freecodecamp.org/news/regex-for-date-formats-what-is-the-regular-expression-for-matching-dates/
        date_spans = [match.span() for match in date_matches]
        phone_matches = re.finditer('[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', text)
        # phone regex is from : https://ihateregex.io/expr/phone/
        phone_spans = [match.span() for match in phone_matches]
        order_matches = re.finditer('\d+\.', text)
        # this is for cases like 2. = second
        order_spans = [match.span() for match in order_matches]
        apostrophe_matches = re.finditer("\w+'\w+", text)
        apostrophe_spans = [match.span() for match in apostrophe_matches]
        spans = email_spans + url_spans + currency_spans + currency_spans_2 + number_spans + date_spans + phone_spans + order_spans + apostrophe_spans

        for span in spans:
            if (span[0] != span[1]):
                for i in range(span[0], span[1]):
                    immunes[i] = 1

        for i in range(len(text)):
            if text[i] == ' ':
                white_space_places.append(i)

        splitted = text.split(" ")

        for j in range(2):
            for i in range(len(splitted)):
                if (j == 0):
                    if (i < len(splitted) - 2):
                        if (" ".join(splitted[i:i + 3]) in self.mwe_lexicon):
                            # print(" ".join(splitted[i:i+3]))
                            if (i == 0):
                                immunes[:white_space_places[i + 1] + len(splitted[i + 2]) + 1] = [1 for k in range(
                                    len(" ".join(splitted[i:i + 3])))]
                            else:
                                immunes[
                                white_space_places[i - 1] + 1:white_space_places[i + 1] + len(splitted[i + 2]) + 1] = [1
                                                                                                                       for
                                                                                                                       k
                                                                                                                       in
                                                                                                                       range(
                                                                                                                           len(" ".join(
                                                                                                                               splitted[
                                                                                                                               i:i + 3])))]
                if (j == 1):
                    if (i < len(splitted) - 1):
                        if (" ".join(splitted[i:i + 2]) in self.mwe_lexicon):
                            # print(" ".join(splitted[i:i+2]))
                            if (i == 0):
                                immunes[:white_space_places[i] + len(splitted[i + 1]) + 1] = [1 for k in range(
                                    len(" ".join(splitted[i:i + 2])))]
                            else:
                                immunes[
                                white_space_places[i - 1] + 1:white_space_places[i] + len(splitted[i + 1]) + 1] = [1 for
                                                                                                                   k in
                                                                                                                   range(
                                                                                                                       len(" ".join(
                                                                                                                           splitted[
                                                                                                                           i:i + 2])))]
        for i in range(len(splitted)):
            if splitted[i] in self.abbreviation_set:
                if (i == 0):
                    immunes[:len(splitted[i])] = [1 for j in range(len(splitted[i]))]
                else:
                    immunes[white_space_places[i - 1] + 1:white_space_places[i]] = [1 for j in range(len(splitted[i]))]

        for i in range(len(text)):
            if text[i] == ' ' or text[i] in punctuations:
                if (immunes[i] == 0):
                    split_points.add(i)

        split_points = list(split_points)
        split_points.sort()
        return split_points

    def tokenize(self, text):
        text = self.preprocess(text)

        split_points = self.find_split_points(text)
        tokens = []
        for i in range(len(split_points)):
            if (i == 0):
                tokens.append(text[:split_points[i]])
                tokens.append(text[split_points[i]])
            else:
                if (split_points[i] > split_points[i - 1] + 1):
                    tokens.append(text[split_points[i - 1] + 1:split_points[i]])
                    tokens.append(text[split_points[i]])
                else:
                    tokens.append(text[split_points[i]])

        if (split_points[-1] < len(text) - 1):
            tokens.append(text[split_points[-1] + 1:])

        final_tokens = []
        for token in tokens:
            if token != " ":
                final_tokens.append(token)
        return final_tokens

    def tokenize_without_punctuations(self, text):
        tokens = self.tokenize(text)
        new_tokens = []
        punctuations = set(
            ["?", "!", "…", ":", ",", ";", ".", '"', "'", "\\", "/", "(", ")", "[", "]", "‘", "’", "“", "”"])
        for token in tokens:
            if token not in punctuations:
                new_tokens.append(token)
        return new_tokens

    def bigram_ready_tokenize(self, text):
        text = self.preprocess(text)
        sentences = SentenceSplitter().rule_based_split(text)
        final_tokens = []
        for sentence in sentences:
            tokens = self.tokenize_without_punctuations(sentence)
            tokens.insert(0, "<s>")
            tokens.append("<\s>")
            final_tokens += tokens
        return final_tokens