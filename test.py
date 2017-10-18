import gensim
import nltk
import random
import codecs
import os
import string
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
random.seed(123)
txt = codecs.open("pg3300.txt", "r", "utf-8").read()

# Removes header and footer
sections = txt.split("***")
mainTxt = sections[2]
# Splits mainTxt into paragraphs
paragraphs = mainTxt.split(2 * os.linesep)

# Tokenizes paragraphs
tokens = []
for para in paragraphs:
    if para:  # Removes blank paragraphs
        for ch in string.punctuation + "\n\r\t":
            para = para.replace(ch, "")
        para = para.lower()
        tokens.append(para.split())
print("Book tokenized")

#Stem tokens
stemmer = nltk.stem.PorterStemmer()
stemmedTokens = []
for i in range(0, len(tokens)):
    para = []
    if tokens[i] is not None:
        for j in range(0, len(tokens[i])):
            para.append(stemmer.stem(tokens[i][j]))
    stemmedTokens.append(para)
print("Finished stemming")

# Creating dictionary
dictionary = gensim.corpora.Dictionary()
dictionary.add_documents(stemmedTokens)
print("Dictionary created")

# Remove stop words
stopWords = codecs.open("common-english-words.txt", "r", "utf-8").read().split(",")
stop_ids = [dictionary.token2id[stopword] for stopword in stopWords
            if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)
dictionary.compactify()
print("Stop words removed")

# Creates Bag of Words
corpus = [dictionary.doc2bow(token) for token in stemmedTokens]

# Create TF-ID- and LSI model and corpus
tfidModel = gensim.models.TfidfModel(corpus)
tfidCorpus = tfidModel[corpus]

# Creates LSI model and corpus
lsiModel = gensim.models.LsiModel(tfidCorpus, id2word=dictionary, num_topics=100)
lsiCorpus = lsiModel[tfidCorpus]

# Query
# query = "Produced by Colin Muir"
# vec_bow = dictionary.doc2bow(query.lower().split())
# vec_lsi = lsi[vec_bow]
# vec_tfid = tfid[query]
# print(vec_tfid)

#Creates similarity models
TFIDsimModel = gensim.similarities.MatrixSimilarity(tfidCorpus)
LSIsimModel = gensim.similarities.MatrixSimilarity(lsiCorpus)

