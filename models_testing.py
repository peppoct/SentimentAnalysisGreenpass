import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from text_normalization import normalize_text
from pre_processing import clening

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

from text_normalization import normalize_text

scoring = [
    'accuracy',
    'recall_weighted',
    'precision_weighted',
    'f1_weighted'
    #'roc_auc'
]

""""
# ------------------------ t-test ------------------------ #
def t_test(pipelineA, pipelineB, dataset, labels, iter):
    resultA = []
    
     #Python paired sample t-test
     ttest_rel(a, b)
    """


def print_score(model, name):

    scores = cross_validate(model, data, labels, cv=10, scoring=scoring)
    fit_time = scores['fit_time']
    accuracy = scores['test_accuracy']
    precision = scores['test_precision_weighted']
    recall = scores['test_recall_weighted']
    f_score = scores['test_f1_weighted']

    print(name + " SCORE:")
    print("Accuracy  : %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))
    print("Precision  : %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))
    print("Recall  : %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))
    print("f measure  : %0.2f (+/- %0.2f)" % (f_score.mean(), f_score.std() * 2))
    #print(scores)

# ------------------------ loading input data ------------------------ #
dataset = pd.read_csv("./dataset/july_to_be_targeted.csv", usecols=['content', 'sentiment'], dtype={'content':'str', 'setiment':'int'})
dataset = dataset[~dataset['sentiment'].isnull()]
#neutral = dataset[dataset['sentiment'] == 0]
#positive = dataset[dataset['sentiment'] == 1]
#negative = dataset[dataset['sentiment'] == -1]
#dataset = pd.concat([neutral, positive, negative])
#neutral = dataset[dataset['sentiment'] == 0.0]

#neutral = neutral.sample(n=200)
#positive = dataset[dataset['sentiment'] == 1.0]
#negative = dataset[dataset['sentiment'] == -1.0]
#dataset = pd.concat([neutral, positive, negative])
dataset = dataset.sample(frac=1)
dataset = clening(dataset)
data = dataset['content']
labels = dataset['sentiment']

data = normalize_text(data)


labels_codes = {
    "positive" : 1,
    "neutral" : 0,
    "negative" : -1
}

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)



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
'''
tuned_parameters_decision_tree = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'fselect__k': ('all', 1000, 2000, 3500),
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': np.arange(1, 10),
    'clf__min_samples_split': np.arange(1,10),
    'clf__min_samples_leaf': np.arange(1,5)
}
'''

decisionTree = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', DecisionTreeClassifier(random_state=0)),
    ])

print_score(decisionTree, 'decision tree')


# --------------- RANDOM FOREST ---------------

randomForest = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', RandomForestClassifier(n_estimators=1200)),
    ])

scores = cross_val_score(randomForest, data, labels, cv=10)
print("Accuracy RandomForest : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- LOGISTIC REGRESSION ---------------

'''
tuned_parameters_svm = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'fselect__k': ('all', 1000, 2000, 3500),
    'clf__penalty': ['l1','l2'],
    'clf__C': [0.001,0.01,0.1,1,10,100,1000]
}
'''

logisticRegression = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', LogisticRegression()),
    ])

scores = cross_val_score(logisticRegression, data, labels, cv=10)
print("Accuracy LogisticRegression : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


# --------------- SVM ---------------

'''
tuned_parameters_svm = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'fselect__k': ('all', 1000, 2000, 3500),
    'clf__C': [0.1, 1, 10, 100, 1000],
    'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'clf__kernel': ['rbf', 'linear']
}
'''

svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', svm.SVC()),
    ])


scores = cross_val_score(svm, data, labels, cv=10)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- NAIVE-BAYES ---------------

'''
tuned_parameters_multinomial = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'fselect__k': ('all', 1000, 2000, 3500),
    'clf__alpha': [1, 1e-1, 1e-2]
}
'''

multinomialNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, K=3500)),
    ('clf', MultinomialNB()),
    ])

scores = cross_val_score(multinomialNB, data, labels, cv=10)
print("Accuracy MultinomialNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)


'''
gaussianNB = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', GaussianNB()),
    ])

scores = cross_val_score(gaussianNB, data, labels, cv=10)
print("Accuracy GaussianNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
'''
# --------------- K-NN ---------------
knn = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', KNeighborsClassifier(n_neighbors=3)),
    ])

scores = cross_val_score(knn, data, labels, cv=10)
print("Accuracy KNN : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- ADABOOST ---------------
adaboost = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', AdaBoostClassifier()),
    ])

scores = cross_val_score(adaboost, data, labels, cv=10)
print("Accuracy ADABOOST : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

# --------------- GBC ---------------
gbc = Pipeline([
    ('vect', stem_vectorizer),
    ('tfidf', TfidfTransformer(smooth_idf=True,use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', GradientBoostingClassifier()),
    ])

scores = cross_val_score(gbc, data, labels, cv=10)
print("Accuracy GBC : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

bg_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', BaggingClassifier(base_estimator=svm.SVC(), n_estimators=1100)),
    ])

scores = cross_val_score(bg_svm, data, labels, cv=10)
print("Accuracy BG-SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)

def best_paramiters_combination(model, tuned_paramiters):
    clf = GridSearchCV(model, tuned_paramiters, cv=10, scoring='accuracy')
    clf.fit(x_train, y_train)
    print("Best configuration MultinominalNB:")
    print(clf.best_params_)
    print(classification_report(y_test, clf.predict(x_test), digits=4))

'''
models_pipelines = [
    {"model name": "Decision Tree", "model": decisionTree},
    {"model name": "Random Forest Classifier", "model": randomForest},
    {"model name": "Logistic Regression", "model": logisticRegression},
    {"model name": "SVM", "model": svm},
    {"model name": "MultinomialNB", "model": multinomialNB},
    {"model name": "KNeighbors", "model": knn},
    {"model name": "Adaboost", "model": adaboost},
    {"model name": "Gradient Boosting Classifier", "model": gbc},
    ]
    '''
"""
# ----------------------- cross test ----------------------- #
for m1 in models_pipelines:
    for m2 in models_pipelines:
        if m1 == m2:
            continue
        print("Testing Models: M1 {} M2 {}".format(m1["model_name"], m2["model_name"]))
        #tTest(m1["model"], m2["model"])"""