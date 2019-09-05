#!/usr/bin/env python
# coding: utf-8

# ## Semantic Word Clustering from Large Arabic Text
# In this Jupyter Notebook, I show a solution of creating semantic word clustering from large Arabic plain unstructured text. This approach uses neural word embedding models “word2vec library” to create word vectors and avoid using one-hot encoding which is not consider the semantic relationships between Arabic words. Also, the beauty of embedding approach is that not much feature engineering is needed. This approach is part of my thesis work for my master degree of information technology in 2018. I will use word2vec google library to create word vectors and then create classification features and finally creates word clusters.

# ## Downloading and Preprocessing Data
# I download the Wikimedia database dump of Arabic Wikipedia on May 20, 2017 (https://archive.org/details/arwiki-20170520) . The volume of Arabic Wikimedia articles has reached as of September 28, 2017 above 510,651 articles (from different domains such as politics, economy, comedy, history and others). The text volume about (1.7GB) and become (1.3GB) after pre-processing. The preprocessing stage consists of dropping the diacritical marks and the character elongation and data normalization (dropping any character and symbols except Arabic characters).
# 

# In[ ]:


# -*- coding: utf-8 -*-
import re
import codecs
import pyarabic.araby as araby
import timeit


def read_file():
    f = codecs.open("3g.txt", "r", encoding='utf8')
    data = f.read()
    f.close()
    return data


def write_file(data):
    filew = codecs.open('3g-p.txt', 'w', encoding='utf8')
    filew.write(data)

    filew.close()


def write_one_col():
    with open('10-wiki.txt', 'r') as f, open('one-col.txt', 'w') as f2:
        for line in f:
            for word in line.split():
                f2.write(word + '\n')


def normailize_data(data):
    regex = ur'[\u0621-\u063A\u0641-\u064A]+'
    return " ".join(re.findall(regex, data))


def strip_tatweel(text):

    reduced = araby.strip_tatweel(text)
    return reduced


def strip_tashkeel(text):
    reduced = araby.strip_tashkeel(text)
    return reduced


start_time = timeit.default_timer()
data = read_file()
remove_tashkeel = strip_tashkeel(data)
remove_tatweel = strip_tatweel(remove_tashkeel)
normailized = normailize_data(remove_tatweel)
write_file(normailized)

elapsed = timeit.default_timer() - start_time
InMinutes = elapsed / 60

print ("The Totatl Execution Time in Minutes is: ", InMinutes)


# ## Creating Word Vectors
# We create word vectors using word2vec model. Two stages are applied to create word vectors. First; word phrases are created to constitute the bigram statistics for the word frequency in text corpus, then they are used as better input for word2vec model. Second; creating the vector representation of words using the CBOW model.
# 

# In[ ]:


from gensim.models.keyedvectors import KeyedVectors
import word2vec
import timeit


start_time = timeit.default_timer()

word2vec.word2phrase('3g-p.txt', '3g-phrases.txt', verbose=True)
word2vec.word2vec('3g-phrases.txt', '3g.bin', size=100, verbose=True)

elapsed = timeit.default_timer() - start_time
InMinutes = elapsed / 60

word2vec.word2clusters('3g-p.txt', '3g-clusters.txt', 100, verbose=True)

model = KeyedVectors.load_word2vec_format('3g.bin', binary=True)

model.save_word2vec_format('3g-vectors.txt', binary=False)

print ("The Totatl Execution Time in Minutes is: ", InMinutes)


# ## Feature Extraction
# Building the classification features for the classification tasks is performed by constructing Term Frequency Inverse Document Matrix (TF-IDF). This matrix scores importance of words or terms in a document based on how frequently they appear across multiple documents. In order to construct TF-IDF matrix from the generated word vectors, we average the word vectors for each word. We used a GitHub repository class (MeanEmbeddingVectorizer) written by Nadbordrozd (Nadbordrozd, 2016)
# 

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import data
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


#Store the word vectors into dictionary
with open("1900m-vectors.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

# build the features, by averaging the word vectors for all vectors in the text
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# ## Building the Classification models
# In order to classify the generated word vectors and create the classification model, we used the Sklearn’s pipeline. The pipeline sequentially applies a list of transforms (such as extracting text documents and tokenizes them) before passing the resulting features along to classifier algorithms. We used the pipeline object to transform the process of averaging word vectors and creating classification features and passes these features to extra tree classifier

# In[ ]:


# classify the vectors using Extra Trees classifier
model = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

xx = data.get_data('train.txt')
yy = data.label_data('label.txt')

X = np.array(xx)
y = np.array([yy]).reshape(150)

kf = KFold(n_splits=10, shuffle=True, random_state=None)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    data_train, data_test = X[train_index], X[test_index]
    target_train, target_test = y[train_index], y[test_index]

model.fit(data_train, target_train)

prediction = model.predict(data_test)
print(pd.DataFrame({'words': data_test, 'prediction': prediction}))

expected = target_test
predicted = model.predict(data_test)

cm = confusion_matrix(target_test, prediction)

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print accuracy_score(target_test, predicted)

print  model_selection.cross_val_score(model, data_train, target_train)

print('the classification Scores:\n')

target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5',
                'class 6', 'class 7', 'class 8', 'class 9', 'class 10',
                'class 11', 'class 12', 'class 13', 'class 14', 'class 15']

print '\nClasification report:\n', classification_report(target_test, prediction, target_names=target_names)

