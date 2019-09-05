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


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


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
