import re
import nltk
from typing import Dict, Set, List
import numpy as np
from tokenizer_rule import RuleBasedTokenizer


class MLBasedTokenizer:
    def __init__(self):
        self.train_corpus, self.test_corpus = self.load_train_test_corpus()
        self.letters = "a,b,c,ç,d,e,f,g,ğ,h,ı,i,j,k,l,m,n,o,ö,p,r,s,ş,t,u,ü,v,y,z,q,w,x".split(",")
        self.upper_letters = [letter.upper() for letter in self.letters]
        self.numbers = [str(i) for i in range(10)]
        self.split_points = RuleBasedTokenizer().find_split_points(self.train_corpus)

    @staticmethod
    def load_train_test_corpus():
        with open("../data/train_corpus.txt", "r", encoding="utf-8-sig") as f:
            train_corpus = f.read()

        with open("../datatest_corpus.txt", "r", encoding="utf-8-sig") as f:
            test_corpus = f.read()

        return RuleBasedTokenizer().preprocess(train_corpus), RuleBasedTokenizer().preprocess(test_corpus)

    def find_category(self, char):
        if char == " ":
            return " "
        if char == ".":
            return "."
        if char in set(self.letters):
            return "letter"
        if char in set(self.upper_letters):
            return "capital"
        if char in self.numbers:
            return "number"
        if char == "":
            return "empty"

        return "rest"

    def train(self, feature_set=1):
        char_list = list(self.train_corpus).copy()
        if (0 < feature_set < 5 == False):
            print("Invalid feature set")
            return

        class_1_prob = np.log(len(self.split_points) / len(self.train_corpus))
        class_0_prob = np.log(1 - (len(self.split_points) / len(self.train_corpus)))
        if feature_set == 1:
            char_list.insert(0, "")
            char_list.append("")
            counts = [[{} for i in range(2)] for j in range(3)]
            cond_probabilities = [[{} for i in range(2)] for j in range(3)]

            for i in range(1, len(char_list) - 1):
                decision = 0
                if (i - 1 in self.split_points):
                    decision = 1

                for j in range(-1, 2):

                    if (char_list[i + j] in counts[j + 1][decision]):
                        counts[j + 1][decision][char_list[i + j]] += 1
                    else:
                        counts[j + 1][decision][char_list[i + j]] = 1

            for i in range(3):
                for j in range(2):
                    total = sum(counts[i][j].values())
                    cond_probabilities[i][j] = {k: v / total for k, v in counts[i][j].items()}

            return cond_probabilities, class_1_prob, class_0_prob

        if feature_set == 2:
            char_list.insert(0, "")
            char_list.insert(0, "")
            char_list.append("")
            char_list.append("")
            counts = [[{} for i in range(2)] for j in range(5)]
            cond_probabilities = [[{} for i in range(2)] for j in range(5)]

            for i in range(2, len(char_list) - 2):
                decision = 0
                if (i - 2 in self.split_points):
                    decision = 1

                for j in range(-2, 3):

                    if (char_list[i + j] in counts[j + 2][decision]):
                        counts[j + 2][decision][char_list[i + j]] += 1
                    else:
                        counts[j + 2][decision][char_list[i + j]] = 1

            for i in range(5):
                for j in range(2):
                    total = sum(counts[i][j].values())
                    cond_probabilities[i][j] = {k: v / total for k, v in counts[i][j].items()}

            return cond_probabilities, class_1_prob, class_0_prob

        if feature_set == 3:
            char_list.insert(0, "")
            char_list.append("")
            counts = [[{} for i in range(2)] for j in range(3)]
            cond_probabilities = [[{} for i in range(2)] for j in range(3)]

            for i in range(1, len(char_list) - 1):
                decision = 0
                if (i - 1 in self.split_points):
                    decision = 1

                for j in range(-1, 2):

                    if (self.find_category(char_list[i + j]) in counts[j + 1][decision]):
                        counts[j + 1][decision][self.find_category(char_list[i + j])] += 1
                    else:
                        counts[j + 1][decision][self.find_category(char_list[i + j])] = 1

            for i in range(3):
                for j in range(2):
                    total = sum(counts[i][j].values())
                    cond_probabilities[i][j] = {k: v / total for k, v in counts[i][j].items()}

            return cond_probabilities, class_1_prob, class_0_prob

        if feature_set == 4:
            char_list.insert(0, "")
            char_list.insert(0, "")
            char_list.append("")
            char_list.append("")

            counts = [[{} for i in range(2)] for j in range(5)]
            cond_probabilities = [[{} for i in range(2)] for j in range(5)]

            for i in range(2, len(char_list) - 2):
                decision = 0
                if (i - 2 in self.split_points):
                    decision = 1

                for j in range(-2, 3):

                    if (self.find_category(char_list[i + j]) in counts[j + 2][decision]):
                        counts[j + 2][decision][self.find_category(char_list[i + j])] += 1
                    else:
                        counts[j + 2][decision][self.find_category(char_list[i + j])] = 1

            for i in range(5):
                for j in range(2):
                    total = sum(counts[i][j].values())
                    cond_probabilities[i][j] = {k: v / total for k, v in counts[i][j].items()}

            return cond_probabilities, class_1_prob, class_0_prob

    def get_preds(self, cond_probabilites, class_1_prob, class_0_prob, document, feature_set=1):
        # model will have the conditional probabilities and the class probabilities
        if (0 < feature_set < 5 == False):
            print("Invalid feature set")
            return

        predictions = []

        char_list = list(document).copy()

        if feature_set == 1:
            char_list.insert(0, "")
            char_list.append("")
            for i in range(1, len(char_list) - 1):
                split_prob = class_1_prob
                no_split_prob = class_0_prob
                for j in range(3):
                    if char_list[i - 1 + j] in cond_probabilites[j][1]:
                        split_prob += np.log(cond_probabilites[j][1][char_list[i - 1 + j]])
                    else:
                        split_prob += -100

                    if char_list[i - 1 + j] in cond_probabilites[j][0]:
                        no_split_prob += np.log(cond_probabilites[j][0][char_list[i - 1 + j]])
                    else:
                        no_split_prob += -100

                if (split_prob > no_split_prob):
                    predictions.append(i - 1)

            return predictions

        if feature_set == 2:
            char_list.insert(0, "")
            char_list.insert(0, "")
            char_list.append("")
            char_list.append("")
            for i in range(2, len(char_list) - 2):
                split_prob = class_1_prob
                no_split_prob = class_0_prob
                for j in range(5):
                    if char_list[i - 2 + j] in cond_probabilites[j][1]:
                        split_prob += np.log(cond_probabilites[j][1][char_list[i - 2 + j]])
                    else:
                        split_prob += -100

                    if char_list[i - 2 + j] in cond_probabilites[j][0]:
                        no_split_prob += np.log(cond_probabilites[j][0][char_list[i - 2 + j]])
                    else:
                        no_split_prob += -100
                if (split_prob > no_split_prob):
                    predictions.append(i - 2)

            return predictions

        if feature_set == 3:
            char_list.insert(0, "")
            char_list.append("")
            for i in range(1, len(char_list) - 1):
                split_prob = class_1_prob
                no_split_prob = class_0_prob
                for j in range(3):
                    if self.find_category(char_list[i - 1 + j]) in cond_probabilites[j][1]:
                        split_prob += np.log(cond_probabilites[j][1][self.find_category(char_list[i - 1 + j])])
                    else:
                        split_prob += -100

                    if self.find_category(char_list[i - 1 + j]) in cond_probabilites[j][0]:
                        no_split_prob += np.log(cond_probabilites[j][0][self.find_category(char_list[i - 1 + j])])
                    else:
                        no_split_prob += -100

                if (split_prob > no_split_prob):
                    predictions.append(i - 1)

            return predictions

        if feature_set == 4:
            char_list.insert(0, "")
            char_list.insert(0, "")
            char_list.append("")
            char_list.append("")
            for i in range(2, len(char_list) - 2):
                split_prob = class_1_prob
                no_split_prob = class_0_prob
                for j in range(5):
                    if self.find_category(char_list[i - 2 + j]) in cond_probabilites[j][1]:
                        split_prob += np.log(cond_probabilites[j][1][self.find_category(char_list[i - 2 + j])])
                    else:
                        split_prob += -100

                    if self.find_category(char_list[i - 2 + j]) in cond_probabilites[j][0]:
                        no_split_prob += np.log(cond_probabilites[j][0][self.find_category(char_list[i - 2 + j])])
                    else:
                        no_split_prob += -100

                # if(i%30 == 0): print(split_prob, no_split_prob)
                if (split_prob > no_split_prob):
                    predictions.append(i - 2)

            return predictions

    def metrics(self, ground_truth, predictions, document_len):
        labels = [0 for i in range(document_len)]
        preds = [0 for i in range(document_len)]

        for split_place in ground_truth:
            labels[split_place] = 1

        for split_place in predictions:
            preds[split_place] = 1

        accuracy = 0
        for i in range(len(preds)):
            if (labels[i] == preds[i]):
                accuracy += 1

        accuracy = accuracy / document_len

        corrects_1 = 0
        corrects_0 = 0
        preds_1 = 0
        preds_0 = 0
        for i in range(len(preds)):
            if preds[i] == 1:
                preds_1 += 1
                if labels[i] == 1:
                    corrects_1 += 1
            if preds[i] == 0:
                preds_0 += 1
                if labels[i] == 0:
                    corrects_0 += 1

        precision_0 = corrects_0 / preds_0
        precision_1 = corrects_1 / preds_1
        recall_0 = corrects_0 / labels.count(0)
        recall_1 = corrects_1 / labels.count(1)

        print(recall_0)
        print(recall_1)
        print(precision_0)
        print(precision_1)
        print(accuracy)

        return

    def tokenize_with_trained_model(self, text, feature_set=1):
        cond_prob, class_1_prob, class_0_prob = self.train(feature_set=feature_set)
        predicted_split_points = self.get_preds(cond_prob, class_1_prob, class_0_prob, text, feature_set=feature_set)
        tokens = []
        for i in range(len(predicted_split_points)):
            if (i == 0):
                tokens.append(text[:predicted_split_points[i]])
                tokens.append(text[predicted_split_points[i]])
            else:
                if (predicted_split_points[i] > predicted_split_points[i - 1] + 1):
                    tokens.append(text[predicted_split_points[i - 1] + 1:predicted_split_points[i]])
                    tokens.append(text[predicted_split_points[i]])
                else:
                    tokens.append(text[predicted_split_points[i]])

        if (self.split_points[-1] < len(text) - 1):
            tokens.append(text[predicted_split_points[-1] + 1:])

        final_tokens = []
        for token in tokens:
            if token != " ":
                final_tokens.append(token)
        return final_tokens