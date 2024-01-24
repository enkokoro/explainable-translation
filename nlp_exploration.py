from typing import List, Tuple
from PyMultiDictionary import MultiDictionary, DICT_WORDNET, DICT_EDUCALINGO
py_multi_dictionary = MultiDictionary()

import deep_translator

import konlpy

# English
import nltk

lemmatizer = nltk.stem.WordNetLemmatizer()
# korean_pos = konlpy.tag._komoran.Komoran()
korean_pos = konlpy.tag._okt.Okt()

def convert_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None

def english(text):
    tokens = nltk.tokenize.word_tokenize(text)
    print(tokens)
    pos_tags = nltk.pos_tag(tokens)
    print(pos_tags)
    base_form = []
    for word, pos in pos_tags:
        c_pos = convert_pos(pos)
        if c_pos:
            base_form.append(lemmatizer.lemmatize(word, pos=c_pos))
        else:
            base_form.append(word)
    print(base_form)

def korean(text):
    pos_tags = korean_pos.pos(text)
    print(pos_tags)
    pass

def translate(text, dest='en', src='auto'):
    return deep_translator.GoogleTranslator(source=src, target=dest).translate(text)
   
english_sample_text = "They refused to permit us to obtain the refuse permit."
english(english_sample_text)

mandarin_sample_text = translate(english_sample_text, dest='zh-TW', src='en')
print(mandarin_sample_text)

japanese_sample_text = translate(english_sample_text, dest='ja', src='en')
print(japanese_sample_text)

korean_sample_text = translate(english_sample_text, dest='ko', src='en')
print(korean_sample_text)
korean(korean_sample_text)

class LanguageTools:
    def __init__(self, language_symbol: str, language_name_english: str, language_name: str):
        self.language_symbol = language_symbol 
        self.language_name_english = language_name_english
        self.language_name = language_name
    def tokenizer(self, text: str) -> List[str]:
        raise NotImplementedError("tokenizer not yet implemented")
    def pos_tagger(self, tokens: List[str]) -> List[Tuple[str,str]]:
        raise NotImplementedError("pos_tagger not yet implemented")
    def lemmatizer(self, tokens_with_tags: List[Tuple[str,str]]) -> List[str]:
        return [self.single_lemmatizer(word, pos) for word, pos in tokens_with_tags]
    def single_lemmatizer(self, word: str, pos) -> str:
        raise NotImplementedError("single_lemmatizer not yet implemented")
    def dictionary(self, tokens_with_tags: List[Tuple[str,str]]) -> List[List[str]]:
        return [self.single_dictionary(word, pos) for word, pos in tokens_with_tags]
    def single_dictionary(self, word: str, pos) -> List[str]:
        raise NotImplementedError("dictionary not yet implemented")
    
class EnglishLanguageTools(LanguageTools):
    def __init__(self):
        LanguageTools.__init__(self, "en", "English", "English")
        nltk.download('punkt') # word_tokenize
        nltk.download('averaged_perceptron_tagger') # pos_tag
        nltk.download('wordnet') # WordNetLemmatizer
    def convert_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return None
    def tokenizer(self, text: str) -> List[str]:
        return nltk.tokenize.word_tokenize(text)
    def pos_tagger(self, tokens: List[str]) -> List:
        return nltk.pos_tag(tokens)
    def single_lemmatizer(self, word: str, pos) -> str:
        pos = self.convert_pos(pos)
        if pos:
            return nltk.stem.WordNetLemmatizer().lemmatize(word, pos)
        else:
            return nltk.stem.WordNetLemmatizer().lemmatize(word)
    def single_dictionary(self, word: str, pos) -> List[str]:
        return py_multi_dictionary.meaning(self.language_symbol, word)
  
class SentenceAnnotation:
    def __init__(self, language: str = 'en'):
        # confirm language is supported and get language specific tools
        language_tools = {
            'en': EnglishLanguageTools
        }
        supported_languages = language_tools.keys()

        if language not in supported_languages:
            raise NotImplementedError("language: " + language + " is not yet supported")
        
        lt = language_tools[language]()
        self.tokenizer   = lt.tokenizer
        self.pos_tagger  = lt.pos_tagger
        self.lemmatizer  = lt.lemmatizer
        self.dictionary  = lt.dictionary

    def annotater(self, text: str):
        # follow general pipeline of annotating sentence
        self.text = text
        self.tokens = self.tokenizer(self.text)
        self.pos_tags = self.pos_tagger(self.tokens)
        self.pos_tag_only = [pos for _, pos in self.pos_tags]
        self.lemmatize = self.lemmatizer(self.pos_tags)
        self.definition = self.dictionary(self.pos_tags)
        # rank possible sense definitions in order of most likely

        self.full_annotation = list(zip(self.tokens, self.pos_tag_only, self.lemmatize, self.definition))
        return self.full_annotation
 
english_annotation = SentenceAnnotation('en')
print(english_annotation.annotater(english_sample_text))