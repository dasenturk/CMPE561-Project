import regex as re
import nltk
from src.tokenizer_rule import RuleBasedTokenizer
from src.sentence_splitter import SentenceSplitter


class Normalizer:
    def __init__(self):
        self.normalization_dicrionary_1, self.normalization_dicrionary_2 = self.create_normalization_dictionaries()
        self.corpus = self.load_corpus()

    @staticmethod
    def create_normalization_dictionaries():
        with open("data/Normalization_Lexicon.txt", "r", encoding="utf-8-sig") as f:
            normalization_lexicon = f.readlines()

        normalization_dictionary_1 = {}

        for line in normalization_lexicon:
            line = line.split("=")
            if (len(line) == 1):
                continue

            candidates = line[1].strip().split(",")
            normalization_dictionary_1[line[0]] = candidates

        with open("data/Normalization_Lexicon_2.txt", "r", encoding="utf-8-sig") as f:
            normalization_lexicon = f.readlines()

        normalization_dictionary_2 = {}

        for line in normalization_lexicon:
            line = line.split("=")
            if (len(line) == 1):
                continue
            correct = line[1].strip()
            normalization_dictionary_2[line[0]] = correct

        return normalization_dictionary_1, normalization_dictionary_2

    @staticmethod
    def load_corpus():
        with open("data/tr_corpus_10M.txt", "r", encoding="utf-8-sig") as f:
            corpus = f.read()
        return corpus

    @staticmethod
    def find_max_index(number_list):
        if (len(number_list) == 0):
            print("no element in the list")
            return

        max_index = 0
        max_value = number_list[0]
        for i in range(len(number_list)):
            if (number_list[i] > max_value):
                max_index = i
                max_value = number_list[i]

        return max_index, max_value

    def normalize(self, document):

        tokens = RuleBasedTokenizer().bigram_ready_tokenize(document)
        bigram_ready_corpus = RuleBasedTokenizer().bigram_ready_tokenize(self.corpus)

        normalized_tokens = []

        for i in range(len(tokens)):
            if (tokens[i] == "<s>" or tokens[i] == "<\s>" or tokens[i] == ""):
                continue

            elif tokens[i] in self.normalization_dicrionary_2:
                normalized_tokens.append(self.normalization_dicrionary_2[tokens[i]])

            elif tokens[i] in self.normalization_dicrionary_1:

                candidate_number = len(self.normalization_dicrionary_1[tokens[i]])
                bigram_with_prev_counts = [0 for l in range(candidate_number)]
                bigram_with_next_counts = [0 for l in range(candidate_number)]
                unigram_prev_count = 0
                unigram_counts = [0 for l in range(candidate_number)]
                bigram_prev_probabilities = [0 for l in range(candidate_number)]
                bigram_next_probabilities = [0 for l in range(candidate_number)]
                bigram_final_probabilities = [0 for l in range(candidate_number)]
                for k in range(candidate_number):
                    for j in range(1, len(bigram_ready_corpus) - 1):
                        if bigram_ready_corpus[j - 1] == tokens[i - 1] and bigram_ready_corpus[j] == tokens[i] and \
                                bigram_ready_corpus[j + 1] == tokens[i + 1]:
                            unigram_prev_count += 1
                            unigram_counts[k] += 1
                            bigram_with_prev_counts[k] += 1
                            bigram_with_next_counts[k] += 1

                        elif bigram_ready_corpus[j - 1] == tokens[i - 1] and bigram_ready_corpus[j] == tokens[i]:
                            unigram_prev_count += 1
                            unigram_counts[k] += 1
                            bigram_with_prev_counts[k] += 1

                        elif bigram_ready_corpus[j] == tokens[i] and bigram_ready_corpus[j + 1] == tokens[i + 1]:
                            unigram_counts[k] += 1
                            bigram_with_next_counts[k] += 1

                        elif bigram_ready_corpus[j - 1] == tokens[i - 1]:
                            unigram_prev_count += 1

                        elif bigram_ready_corpus[j] == tokens[i]:
                            unigram_counts[k] += 1
                    if (unigram_prev_count == 0):
                        bigram_prev_probabilities[k] = 0
                    else:
                        if (bigram_with_prev_counts[k] == 0):
                            bigram_prev_probabilities[k] = 0.00001
                        else:
                            bigram_prev_probabilities[k] = bigram_with_prev_counts[k] / unigram_prev_count

                    if (unigram_counts[k] == 0):
                        bigram_next_probabilities[k] = 0
                    else:
                        if (bigram_with_next_counts[k] == 0):
                            bigram_next_probabilities[k] = 0.00001
                        else:
                            bigram_next_probabilities[k] = bigram_with_next_counts[k] / unigram_counts[k]

                    bigram_final_probabilities[k] = bigram_prev_probabilities[k] * bigram_next_probabilities[k]

                best_candidate, value = self.find_max_index(bigram_final_probabilities)
                if value != 0:
                    normalized_tokens.append(self.normalization_dicrionary_1[tokens[i]][best_candidate])
                    
                else:
                    best_candidate, value = self.find_max_index(unigram_counts)
                    normalized_tokens.append(self.normalization_dicrionary_1[tokens[i]][best_candidate])

            else:
                normalized_tokens.append(tokens[i])
        return normalized_tokens
