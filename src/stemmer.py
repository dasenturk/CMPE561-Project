import re


class Stemmer:
    def __init__(self):
        self.suffixes = [
            # Nominal verb suffixes and Noun suffixes
            '(y)Um', 'sUn', '(y)Uz', 'sUnUz', 'lAr', 'md', 'n', 'k', 'nUz', 'DUr',
            'cAsInA', '(y)DU', '(y)sA', '(y)mUş', '(y)ken',
            '(U)m', '(U)mUz', '(U)n', '(U)nUz', '(s)U', 'lArI', '(y)U', 'nU',
            '(n)Un', '(y)A', 'nA', 'DA', 'nDA', 'DAn', 'nDAn', '(y)lA', 'ki', '(n)cA',
            # Derivational suffixes
            'lUk', 'CU', 'CUk', 'lAş', 'lA', 'lAn', 'CA', 'lU', 'sUz'
        ]

        self.suffixes.sort(key=len, reverse=True)
        self.suffix_regex = self.compile_suffix_regex()

    def compile_suffix_regex(self):
        suffix_patterns = [suffix.replace('U', '[ıiuü]').replace('A', '[ae]').replace('C', '[cç]').replace('D', '[dt]').replace('I', '[ıI]')
                           for suffix in self.suffixes]
        suffix_patterns = [suffix.replace('(y)', '(y)?').replace('(U)', '(U)?').replace('(s)', '(s)?').replace('(n)', '(n)?')
                           for suffix in suffix_patterns]
        pattern = '(' + '|'.join(suffix_patterns) + ')$'
        return re.compile(pattern, re.IGNORECASE)

    def stem(self, word):
        original_word = word
        while True:
            word_before = word
            word = self.suffix_regex.sub('', word)
            if word == word_before or len(word) <= 2:
                break

        if len(word) > 1 and word[-1] in 'bcdğBCDĞ':
            word = word[:-1] + {'b': 'p', 'c': 'ç', 'd': 't', 'ğ': 'k', 'B': 'P', 'C': 'Ç', 'D': 'T', 'Ğ': 'K'}.get(
                word[-1], word[-1])

        return word if len(word) > 2 else word_before

