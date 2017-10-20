import gensim
import nltk
import random
import codecs
import os
import string
import logging
import heapq
import numpy

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
random.seed(123)

# Removes header and footer
def removeHeadAndFoot(txt):
    sections = txt.split("***")
    mainSection = sections[2]
    return mainSection

def splitIntoParagraphs(txt):
    # Splits mainTxt into paragraphs
    paragraphs2 = []
    paragraphs = txt.split(2 * os.linesep)
    for para in paragraphs:
        if para:
            paragraphs2.append(para)
    return paragraphs2

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

# Stem tokens
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

# Creates dictionary
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
def createTFIDmodelsAndCorppus(BoW):
    tfidModel = gensim.models.TfidfModel(BoW)
    return tfidModel

# Creates LSI model and corpus
def createLSImodelsAndCorppus(tfidCorpus, dictionary):
    lsiModel = gensim.models.LsiModel(tfidCorpus, id2word=dictionary, num_topics=100)
    lsiCorpus = lsiModel[tfidCorpus]
    return lsiModel, lsiCorpus

# Uses multiple other functions to preprocess a text
def preprocess(txt):
    paragraphs = splitIntoParagraphs(txt)
    tokens = tokenize(paragraphs)
    stemmedTokens = stem(tokens)
    dictionary = createDictionary(stemmedTokens)
    dictionary = removeStopWords(dictionary)
    return stemmedTokens, dictionary, paragraphs

# Creates TFID and LSI models and corpuses
def createModels(stemmedTokens, dictionary):
    BoW = createBoW(stemmedTokens)
    tfidModel = createTFIDmodelsAndCorppus(BoW)
    #lsiModel, lsiCorpus = createLSImodelsAndCorppus(tfidCorpus, dictionary)
    return BoW, tfidModel #lsiModel, lsiCorpus

#Creates similarity models
def createSimilarity(corpus):
    similarity = gensim.similarities.MatrixSimilarity(corpus)
    return similarity


txt = codecs.open("pg3300.txt", "r", "utf-8").read()
txt = removeHeadAndFoot(txt)

stemmedTokens, dictionary, paragraphs = preprocess(txt)
BoW, tfidModel = createModels(stemmedTokens, dictionary)
tfidCorpus = tfidModel[BoW]
TFIDsimModel = createSimilarity(tfidCorpus)

query = "How taxes influence Economics?"
stemmedTokensQuery, dictionaryQuery, paragraphsQuery = preprocess(query)
BoWQuery, tfidModelQuery = createModels(stemmedTokensQuery, dictionary)

queryTFIDvec = tfidModel[BoWQuery]

index = gensim.similarities.MatrixSimilarity(tfidModel[BoW])
sims = index[queryTFIDvec]

retrievedParagraphs = heapq.nlargest(3, sims[0])
print(numpy.where(sims))

for para in retrievedParagraphs:
    print("\n")
    print(paragraphs[para[0]])
    print("\n")
