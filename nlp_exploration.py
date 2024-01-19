import deep_translator

import konlpy
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
