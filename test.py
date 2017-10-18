import gensim
import nltk
import random
import codecs
import os
import string
import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
random.seed(123)

# Removes header and footer
def removeHeadAndFoot(txt):
    sections = txt.split("***")
    mainSection = sections[2]
    return mainSection

def splitIntoParagraphs(txt):
    # Splits mainTxt into paragraphs
    paragraphs = txt.split(2 * os.linesep)
    return paragraphs

# Tokenizes paragraphs
def tokenize(paragraphs):
    tokens = []
    for para in paragraphs:
        if para:  # Removes blank paragraphs
            for ch in string.punctuation + "\n\r\t":
                para = para.replace(ch, "")
            para = para.lower()
            tokens.append(para.split())
    print("Book tokenized")
    return tokens

#Stem tokens
def stem(tokens):
    stemmer = nltk.stem.PorterStemmer()
    stemmedTokens = []
    for i in range(0, len(tokens)):
        para = []
        if tokens[i] is not None:
            for j in range(0, len(tokens[i])):
                para.append(stemmer.stem(tokens[i][j]))
        stemmedTokens.append(para)
    print("Finished stemming")
    return stemmedTokens

# Creating dictionary
def createDictionary(stemmedTokens):
    dictionary = gensim.corpora.Dictionary()
    dictionary.add_documents(stemmedTokens)
    print("Dictionary created")
    return dictionary

# Remove stop words
def removeStopWords(dictionary):
    stopWords = codecs.open("common-english-words.txt", "r", "utf-8").read().split(",")
    stop_ids = [dictionary.token2id[stopword] for stopword in stopWords
                if stopword in dictionary.token2id]
    dictionary.filter_tokens(stop_ids)
    dictionary.compactify()
    print("Stop words removed")
    return dictionary

# Creates Bag of Words
def createBoW(stemmedTokens):
    BoW = [dictionary.doc2bow(token) for token in stemmedTokens]
    return BoW

# Create TF-ID- and LSI model and corpus
def createTFIDmodelsAndCorppus(BoW, dictionary):
    tfidModel = gensim.models.TfidfModel(BoW)
    tfidCorpus = tfidModel[BoW]
    return tfidModel, tfidCorpus

# Creates LSI model and corpus
def createLSImodelsAndCorppus(tfidCorpus, dictionary):
    lsiModel = gensim.models.LsiModel(tfidCorpus, id2word=dictionary, num_topics=100)
    lsiCorpus = lsiModel[tfidCorpus]
    return lsiModel, lsiCorpus

def preprocess(txt):
    paragraphs = splitIntoParagraphs(txt)
    tokens = tokenize(paragraphs)
    stemmedTokens = stem(tokens)
    dictionary = createDictionary(stemmedTokens)
    dictionary = removeStopWords(dictionary)
    return stemmedTokens, dictionary

def createModels(stemmedTokens, dictionary):
    BoW = createBoW(stemmedTokens)
    tfidModel, tfidCorpus = createTFIDmodelsAndCorppus(BoW, dictionary)
    lsiModel, lsiCorpus = createTFIDmodelsAndCorppus(tfidCorpus, dictionary)
    return BoW, tfidModel, tfidCorpus, lsiModel, lsiCorpus

#Creates similarity models
def createSimMOdels(tfidCorpus, lsiCorpus):
    TFIDsimModel = gensim.similarities.MatrixSimilarity(tfidCorpus)
    LSIsimModel = gensim.similarities.MatrixSimilarity(lsiCorpus)
    return TFIDsimModel, LSIsimModel


txt = codecs.open("pg3300.txt", "r", "utf-8").read()
txt = removeHeadAndFoot(txt)

stemmedTokens, dictionary = preprocess(txt)
BoW, tfidModel, tfidCorpus, lsiModel, lsiCorpus = createModels(stemmedTokens, dictionary)
TFIDsimModel, LSIsimModel = createSimMOdels(tfidCorpus, lsiCorpus)

query = "How taxes influence Economics?"
stemmedTokensQuery, dictionaryQuery = preprocess(query)
BoWQuery, tfidModelQuery, tfidCorpusQuery, lsiModelQuery, lsiCorpusQuery = createModels(stemmedTokensQuery, dictionaryQuery)

print(dictionary.get(2826))
print(lsiCorpusQuery)
print(BoWQuery)