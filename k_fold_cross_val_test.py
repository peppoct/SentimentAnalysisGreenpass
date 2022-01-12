
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import utils
from pre_processing import clening

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from text_normalization import normalize_text

dataset = pd.read_csv("./dataset/prova.csv", usecols=['content', 'sentiment', 'date'], dtype={'content': 'str', 'sentiment': 'int'})
dataset = dataset[~dataset['sentiment'].isnull()]
utils.save_dataset(dataset, 'july_to_be_targeted')

""""
# ------------------------ t-test ------------------------ #
def t_test(pipelineA, pipelineB, dataset, labels, iter):
    resultA = []

     #Python paired sample t-test
     ttest_rel(a, b)
"""

# ------------------------ loading input data ------------------------ #
dataset = pd.read_csv("./dataset/july_to_be_targeted.csv", usecols=['content', 'sentiment'],
                      dtype={'content': 'str', 'sentiment': 'int'})
dataset = dataset[~dataset['sentiment'].isnull()]
dataset = dataset.sample(frac=1)
dataset = clening(dataset)
data = dataset['content']
labels = dataset['sentiment']

#data = normalize_text(data)

labels_codes = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

stop_words = set(stopwords.words('italian'))
stemmer = SnowballStemmer('italian')


class StemmedCountVectorizer(CountVectorizer):
    def __init__(self, stemmer, stop_words):
        super(StemmedCountVectorizer, self).__init__(stop_words=stop_words)
        self.stemmer = stemmer

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))


stem_vectorizer = StemmedCountVectorizer(stemmer, stop_words)

# --------------- DECISION TREE ---------------

decisionTree = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', DecisionTreeClassifier()),
])

scores = cross_val_score(decisionTree, data, labels, cv=10)
print("Accuracy Decision Tree : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- RANDOM FOREST ---------------

randomForest = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', RandomForestClassifier()),
])

scores = cross_val_score(randomForest, data, labels, cv=10)
print("Accuracy RandomForest : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- LOGISTIC REGRESSION ---------------

logisticRegression = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', LogisticRegression()),
])

scores = cross_val_score(logisticRegression, data, labels, cv=10)
print("Accuracy LogisticRegression : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- SVM ---------------

svm = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', svm.SVC()),
])

scores = cross_val_score(svm, data, labels, cv=10)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- NAIVE-BAYES ---------------

multinomialNB = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', MultinomialNB()),
])

scores = cross_val_score(multinomialNB, data, labels, cv=10)
print("Accuracy MultinomialNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


complemntNB = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', ComplementNB()),
    ])

scores = cross_val_score(complemntNB, data, labels, cv=10)
print("Accuracy ComplementNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- ADABOOST ---------------
adaboost = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', AdaBoostClassifier()),
])

scores = cross_val_score(adaboost, data, labels, cv=10)
print("Accuracy ADABOOST : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- GBC ---------------
gbc = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', GradientBoostingClassifier()),
])

scores = cross_val_score(gbc, data, labels, cv=10)
print("Accuracy Gradient Boosting : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- Bagging ---------------

bg = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', BaggingClassifier(n_estimators=100))
])

scores = cross_val_score(bg, data, labels, cv=10)
print("Accuracy BG : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

bg_lr = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', BaggingClassifier(base_estimator=LogisticRegression()))
])

scores = cross_val_score(bg_lr, data, labels, cv=10)
print("Accuracy BG-LR : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

bg_svm = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', BaggingClassifier(base_estimator=svm.SVC()))
])

scores = cross_val_score(bg_svm, data, labels, cv=10)
print("Accuracy BG-SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
