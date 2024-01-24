import nltk

nltk.download('punkt') # word_tokenize
nltk.download('averaged_perceptron_tagger') # pos_tag
nltk.download('wordnet') # WordNetLemmatizer

lemmatizer = nltk.stem.WordNetLemmatizer()
# input: sentence
# tokenization
# part of speech tagging: WordNet
# sense definition: WordSenseDisambiguation
# lemmatizer
# other language possibilities

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

class SentenceAnnotation:
    def __init__(self, text: str):
        self.text = text
        self.tokens = nltk.tokenize.word_tokenize(self.text)
        self.pos_tags = nltk.pos_tag(self.tokens)
        self.pos_tag_only = []
        self.lemmatize = []
        self.synset = []
        self.sense_definition = []
        # rank possible sense definitions in order of most likely

        for word, pos in self.pos_tags:
            self.pos_tag_only.append(pos)
            c_pos = convert_pos(pos)
            if c_pos:
                lemma = lemmatizer.lemmatize(word, pos=c_pos)
                synset = nltk.wsd.lesk(self.tokens, word, c_pos)
            else:
                lemma = word
                synset = nltk.wsd.lesk(self.tokens, word)
            
            definition = None 
            if synset:
                definition = synset.definition()

            self.lemmatize.append(lemma)
            self.synset.append(synset)
            self.sense_definition.append(definition)

        self.full_annotation = list(zip(self.tokens, self.pos_tag_only, self.lemmatize, self.synset, self.sense_definition))

# SentenceAnnotation("I used to live in Pennsylvania")