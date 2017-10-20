import gensim
import nltk
import codecs
import os
import string
import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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

# Prints top 3 paragraph matches for a query
def printTop3Paragraphs(sims):
    for i in range(0, 3):
        print("[", sims[i][0], "]")
        print(paragraphs[sims[i][0]], "\n")


txt = codecs.open("pg3300.txt", "r", "utf-8").read()
txt = removeHeadAndFoot(txt)
paragraphs = splitIntoParagraphs(txt)
tokens = tokenize(paragraphs)
stemmedTokens = stem(tokens)
dictionary = removeStopWords(createDictionary(stemmedTokens))
corpus = [dictionary.doc2bow(token) for token in stemmedTokens]  # Creates a corpus (represented as a Bag of Words)
                                                                 # from the book that can be used to train models

# QUERY
query = "How taxes influence Economics?"
queryTokens = tokenize([query])
queryStemmedTokens = stem(queryTokens)
vec_bow = [dictionary.doc2bow(token) for token in queryStemmedTokens]  # Creates a vector space for the query


#TF-IDF
tfidfModel = gensim.models.TfidfModel(corpus)  # Initialise TF-ID model used for transformation.
                                               # Takes a BoW representation and transforms it to TF-ID weights
corpus_tfidf = tfidfModel[corpus]
vec_tfidf = tfidfModel[vec_bow]  # Transforms the query from BoW -> TF-IDF
for vec in vec_tfidf:
    print(dictionary.get(vec[0][0]), " :", vec[0][1])
    print(dictionary.get(vec[1][0]), " :", vec[1][1])
    print(dictionary.get(vec[2][0]), " :", vec[2][1])
index = gensim.similarities.MatrixSimilarity(corpus_tfidf)  # Indexes the TF-IDF space
sims = index[vec_tfidf]  # Calculates the similarities between the query and the corpus
sims = sorted(enumerate(sims[0]), key=lambda item: -item[1])  # Sorts the simularites by decending order and adds the index
printTop3Paragraphs(sims)

# LSI
lsiModel = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)  # initialize an LSI transformation model
                                                                                     # from the TFIDF corpus
corpus_lsi = lsiModel[corpus_tfidf]
vec_lsi = lsiModel[vec_tfidf]
index = gensim.similarities.MatrixSimilarity(corpus_lsi)  # Indexes the LSI space
sims = index[vec_lsi]  # Calculates the similarities between the query and the corpus
sims = sorted(enumerate(sims[0]), key=lambda item: -item[1])  # Sorts the simularites by decending order and adds the index
printTop3Paragraphs(sims)

print(sorted(vec_lsi, key=lambda kv: -abs(kv[1]))[:3])

