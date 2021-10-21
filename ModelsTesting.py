import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from scipy.stats import ttest_rel

from sklearn.model_selection import cross_val_score, cross_val_predict

""""
# ------------------------ t-test ------------------------ #
def t_test(pipelineA, pipelineB, dataset, labels, iter):
    resultA = []
    
     #Python paired sample t-test
     ttest_rel(a, b)
    """


# ------------------------ loading input data ------------------------ #
dataset = pd.read_csv("./dataset/training_set_july_2021_first_2_weeks.csv")

labels_codes = {
    "positive" : 1,
    "neutral" : 0,
    "negative" : -1
}


stop_words = set(stopwords.words('italian'))
stemmer = SnowballStemmer('italian')

class StemmedCountVectorizer(CountVectorizer):
    def __init__(self, stemmer, stop_words):
        super(StemmedCountVectorizer, self).__init__(stop_words=stop_words)
        self.stemmer = stemmer

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc:(self.stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = StemmedCountVectorizer(stemmer, stop_words)

decisionTree = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=1000)),
    ('clf', DecisionTreeClassifier(random_state=0)),
    ])

scores = cross_val_score(decisionTree, dataset.content, dataset.sentiment, cv=10)
print("Accuracy DecisionTree : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

bg_svm = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=29)),
    ('clf', BaggingClassifier(base_estimator=svm.SVC(), n_estimators=30)),
    ])

scores = cross_val_score(bg_svm, dataset.content, dataset.sentiment, cv=10)
print("Accuracy BG-SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

randomForest = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=4000)),
    ('clf', RandomForestClassifier(criterion="gini", min_samples_split=10, n_estimators=1200)),
    ])

scores = cross_val_score(randomForest, dataset.content, dataset.sentiment, cv=10)
print("Accuracy RandomForest : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

logisticRegression = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', LogisticRegression(C=1, max_iter=1500)),
    ])

scores = cross_val_score(logisticRegression, dataset.content, dataset.sentiment, cv=10)
print("Accuracy LogisticRegression : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

svm = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', svm.SVC(C=10, gamma=0.1)),
    ])

scores = cross_val_score(svm, dataset.content, dataset.sentiment, cv=10)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

knn = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', KNeighborsClassifier(n_neighbors=3)),
    ])

scores = cross_val_score(knn, dataset.content, dataset.sentiment, cv=10)
print("Accuracy KNN : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

models_pipelines = [
    {"model name": "DecisisionTree", "model": decisionTree},
    {"model name": "Bagging + SVM", "model": bg_svm},
    {"model name": "RandomForestClassifier", "model": randomForest},
    {"model name": "LogisticRegression", "model": logisticRegression},
    {"model name": "SVM", "model": svm},
    {"model name": "KNeighbors", "model": knn}
    ]
"""
# ----------------------- cross test ----------------------- #
for m1 in models_pipelines:
    for m2 in models_pipelines:
        if m1 == m2:
            continue
        print("Testing Models: M1 {} M2 {}".format(m1["model_name"], m2["model_name"]))
        #tTest(m1["model"], m2["model"])"""