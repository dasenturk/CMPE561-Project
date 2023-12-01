from src.stemmer import Stemmer
from src.sentence_splitter import SentenceSplitter
from src.tokenizer_rule import RuleBasedTokenizer
#from src.tokenizer_ml import MLBasedTokenizer
from src.tokenizer_ml import tokenize_with_trained_model
from src.normalizer import Normalizer
from src.stopword_eliminator import StopwordEliminator


if __name__ == '__main__':
    with open("data/test_document.txt", "r", encoding="utf-8-sig") as f:
        longer_corpus = f.read()
    
    longer_corpus = RuleBasedTokenizer.preprocess(longer_corpus)
    
    print('------Sentence Splitting------')
    print('--Rule Based--')
    text = SentenceSplitter().rule_based_split(longer_corpus)
    
    print(text)
    """print('--ML Based--')
    text = SentenceSplitter().ml_based_split(longer_corpus)
    print(text)
    """
    print('------Tokenization------')
    print('--Rule Based--')
    text = RuleBasedTokenizer().tokenize(longer_corpus)
    print(text)
    
    print('--ML Based--')
    
    #text = MLBasedTokenizer().tokenize_with_trained_model(longer_corpus)
    text = tokenize_with_trained_model(longer_corpus, feature_set = 1)
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
    
    text= RuleBasedTokenizer().tokenize_without_punctuations(longer_corpus, lower = False)
    stems = [Stemmer().stem(word) for word in text]
    print(stems)



