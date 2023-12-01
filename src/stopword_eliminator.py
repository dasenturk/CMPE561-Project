import re
import nltk
import math
from tokenizer_rule import RuleBasedTokenizer
from sentence_splitter import SentenceSplitter


class StopwordEliminator:
    def __init__(self):
        self.static_stopwords, self.dynamic_stopwords = self.load_stopwords()

    @staticmethod
    def load_stopwords():
        with open("stopwords.txt", "r", encoding="utf-8-sig") as f:
            static_stopwords = f.read()

        with open("dynamic_stopword_list.txt", "r", encoding="utf-8-sig") as f:
            dynamic_stopwords = f.read()

        return set(static_stopwords.split("\n")), set(dynamic_stopwords.split("\n"))

    def remove_stopwords(self, document, stopword_type):
        list_of_tokens = RuleBasedTokenizer().tokenize(document)
        stopword_removed = []
        stopwords = self.static_stopwords if stopword_type == 'static' else self.dynamic_stopwords
        for token in list_of_tokens:
            if token.lower() not in stopwords:
                stopword_removed.append(token)

        return stopword_removed

    def find_stopwords_dynamically(self, corpus): # assume we have a list of documents as corpus
        sentence_splitted_corpus = [SentenceSplitter().rule_based_split(document) for document in corpus]
        word_document_freqs = {}
        word_sentence_freqs = {}
        word_freqs = {}
        scores = {}
        total_sentences = 0
        total_documents = len(corpus)
        for sentence_list in sentence_splitted_corpus:
            tokenized_sentences = [RuleBasedTokenizer().tokenize(sentence) for sentence in sentence_list]
            seen_tokens_document = set()
            total_sentences += len(sentence_list)
            for tokens in tokenized_sentences:
                seen_tokens_sentence = set()
                for token in tokens:
                    if token not in word_freqs:
                        word_freqs[token] = 1
                    else:
                        word_freqs[token] += 1

                    if token not in word_sentence_freqs:
                        word_sentence_freqs[token] = 1

                    else:
                        if token not in seen_tokens_sentence:
                            word_sentence_freqs[token] += 1

                    if token not in word_document_freqs:
                        word_document_freqs[token] = 1

                    else:
                        if token not in seen_tokens_document:
                            word_document_freqs[token] += 1

                    seen_tokens_sentence.add(token)
                    seen_tokens_document.add(token)

        for word in word_freqs:
            if word_document_freqs[word] == 1:
                scores[word] = 0

            else:
                scores[word] = word_freqs[word] * (1 / math.log(total_sentences / (word_sentence_freqs[word] - 1))) * (
                            1 / (math.log(total_documents / (word_document_freqs[word] - 1))))
                # scores[word] = word_freqs[word]*(1/(math.log(total_documents/(word_document_freqs[word]-1))))
                # scores[word] = word_freqs[word]*(1/(math.log(total_documents/(word_sentence_freqs[word]-1))))
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_scores
