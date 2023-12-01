from src.stemmer import Stemmer
from src.sentence_splitter import SentenceSplitter
from src.tokenizer_rule import RuleBasedTokenizer
from src.tokenizer_ml import tokenize_with_trained_model
from src.normalizer import Normalizer
from src.stopword_eliminator import StopwordEliminator


if __name__ == '__main__':
    with open("data/test_document.txt", "r", encoding="utf-8-sig") as f:
        test_document = f.read()
    
    test_document = RuleBasedTokenizer.preprocess(test_document)
    
    print('------Sentence Splitting------')
    print('--Rule Based--')
    text = SentenceSplitter().rule_based_split(test_document)
    
    print(text)
    """print('--ML Based--')
    text = SentenceSplitter().ml_based_split(test_document)
    print(text)
    """
    print('------Tokenization------')
    print('--Rule Based--')
    text = RuleBasedTokenizer().tokenize(test_document)
    print(text)
    
    print('--ML Based--')
    
    #text = MLBasedTokenizer().tokenize_with_trained_model(test_document)
    text = tokenize_with_trained_model(test_document, feature_set = 1)
    print(text)
    
    print('------Stopword Elimination------')
    print('--Static--')
    text = StopwordEliminator().remove_stopwords(test_document, stopword_type='static')
    print(text)
    print('--Dynamic--')
    text = StopwordEliminator().remove_stopwords(test_document, stopword_type='dynamic')
    print(text)

    print('------Normalization------')
    text = Normalizer().normalize(test_document)
    print(text)

    print('------Stemming------')
    
    text= RuleBasedTokenizer().tokenize_without_punctuations(test_document, lower = False)
    stems = [Stemmer().stem(word) for word in text]
    print(stems)



