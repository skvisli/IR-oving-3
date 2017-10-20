import gensim
import nltk
import random
import codecs
import os
import string
import logging
import heapq
import numpy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
def createCorpus(stemmedTokens):
    corpus = [dictionary.doc2bow(token) for token in stemmedTokens]
    return corpus

# Create TF-IDF model
def createTFIDFmodel(corpus):
    tfidfModel = gensim.models.TfidfModel(corpus)
    return tfidfModel

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

# Creates TF-IDF and LSI models and corpuses
def createModels(stemmedTokens, dictionary):
    corpus = createCorpus(stemmedTokens)
    tfidfModel = createTFIDmodel(corpus)
    return corpus, tfidfModel #lsiModel, lsiCorpus

#Creates similarity models
def createSimilarity(corpus):
    similarity = gensim.similarities.MatrixSimilarity(corpus)
    return similarity


txt = codecs.open("pg3300.txt", "r", "utf-8").read()
txt = removeHeadAndFoot(txt)

stemmedTokens, dictionary, paragraphs = preprocess(txt)
corpus = [dictionary.doc2bow(token) for token in stemmedTokens]  # Creates a corpus (represented as a Bag of Words)from the book that can be used to train models
tfidfModel = gensim.models.TfidfModel(corpus)  # Initialise TF-ID model used for transformation. Takes a BoW representation and transforms it to TF-ID weights
corpus_tfidf = tfidfModel[corpus]
lsiModel = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)  # initialize an LSI transformation model
corpus_lsi = lsiModel[corpus_tfidf]
lsiModel.print_topics(2)
#for para in corpus_lsi:
#    print(para)

query = "How taxes influence Economics?"
queryTokens = tokenize([query])
queryStemmedTokens = stem(queryTokens)
vec_bow = [dictionary.doc2bow(token) for token in queryStemmedTokens]
vec_lsi = lsiModel[vec_bow]
#for vec in vec_lsi:
#    print(vec)

index = gensim.similarities.MatrixSimilarity(lsiModel[corpus])  # transform corpus to LSI space and index it
sims = index[vec_lsi]  # perform a similarity query against the corpus
sims = sorted(enumerate(sims[0]), key=lambda item: -item[1])  # Sorts sim decending on the similarity scores
"""print(sims[0])  # print sorted (document number, similarity score) 2-tuples
paragraaa = corpus.__getitem__(sims[0][0])
for para in paragraaa:
    print(dictionary.__getitem__(para[0]))
print(paragraphs[sims[0][0]])
print(paragraphs[sims[1][0]])
print(paragraphs[sims[2][0]])
"""

vec_tfidf = tfidfModel[vec_bow]
for vec in vec_tfidf:
    print(dictionary.get(vec[0][0]), " :", vec[0][1])
    print(dictionary.get(vec[1][0]), " :", vec[1][1])
    print(dictionary.get(vec[2][0]), " :", vec[2][1])
index = gensim.similarities.MatrixSimilarity(tfidfModel[corpus])
sims = index[vec_tfidf]
sims = sorted(enumerate(sims[0]), key=lambda item: -item[1])
print(sims[0])
print(paragraphs[sims[0][0]])
print(paragraphs[sims[1][0]])
print(paragraphs[sims[2][0]])

"""
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
"""