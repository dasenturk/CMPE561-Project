from src.stemmer import Stemmer
from src.sentence_splitter import SentenceSplitter
from src.tokenizer_rule import RuleBasedTokenizer
from src.tokenizer_ml import MLBasedTokenizer
from src.normalizer import Normalizer
from src.stopword_eliminator import StopwordEliminator


if __name__ == '__main__':
    with open("data/test_corpus.txt", "r", encoding="utf-8-sig") as f:
        longer_corpus = f.read()

    print('------Sentence Splitting------')
    print('--Rule Based--')
    text = SentenceSplitter().rule_based_split(longer_corpus)
    print(text)
    print('--ML Based--')
    text = SentenceSplitter().ml_based_split(longer_corpus)
    print(text)

    print('------Tokenization------')
    print('--Rule Based--')
    text = RuleBasedTokenizer().bigram_ready_tokenize(longer_corpus)
    print(text)
    print('--ML Based--')
    text = MLBasedTokenizer().tokenize_with_trained_model(longer_corpus)
    print(text)

    print('------Stopword Elimination------')
    print('--Static--')
    text = StopwordEliminator().remove_stopwords(longer_corpus, stopword_type='static')
    print(text)
    print('--Dynamic--')
    text = StopwordEliminator().remove_stopwords(longer_corpus, stopword_type='dynamic')
    print(text)

    print('------Normalization------')
    text = Normalizer().normalize(longer_corpus)
    print(text)

    print('------Stemming------')
    text = SentenceSplitter().rule_based_split(longer_corpus)
    stems = [Stemmer().stem(word) for word in text]
    print(stems)



