import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pre_processing import clening

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from text_normalization import normalize_text

""""
# ------------------------ t-test ------------------------ #
def t_test(pipelineA, pipelineB, dataset, labels, iter):
    resultA = []

     #Python paired sample t-test
     ttest_rel(a, b)
    """

'''
def print_score(model, name):

    scores = cross_validate(model, data, labels, cv=10, scoring=scoring)
    fit_time = scores['fit_time']
    accuracy = scores['test_accuracy']
    precision = scores['test_precision_weighted']
    recall = scores['test_recall_weighted']
    f_score = scores['test_f1_weighted']


    print("\n----------------------------------------------------------------------------")
    print(name.upper() + " SCORE:")
    print("Accuracy  : %0.2f (+/- %0.4f)" % (accuracy.mean(), accuracy.std() * 2))
    print("Precision  : %0.2f (+/- %0.4f)" % (precision.mean(), precision.std() * 2))
    print("Recall  : %0.2f (+/- %0.4f)" % (recall.mean(), recall.std() * 2))
    print("f measure  : %0.2f (+/- %0.4f)" % (f_score.mean(), f_score.std() * 2))
    #print(scores)
    print("----------------------------------------------------------------------------\n")
'''
# ------------------------ loading input data ------------------------ #
dataset = pd.read_csv("./dataset/july_to_be_targeted.csv", usecols=['content', 'sentiment'],
                      dtype={'content': 'str', 'setiment': 'int'})
dataset = dataset[~dataset['sentiment'].isnull()]
dataset = dataset.sample(frac=1)
dataset = clening(dataset)

data = dataset['content']
labels = dataset['sentiment']

data = normalize_text(data)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='micro', labels=labels, zero_division=True),
    'recall': make_scorer(recall_score, average='micro', labels=labels, zero_division=True),
    'f1_score': make_scorer(f1_score, average='micro', labels=labels, zero_division=True)
    # 'roc_auc': make_scorer(roc_auc_score, average='micro', labels=labels)
}

labels_codes = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

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

# ------------------------------------- SVM ------------------------------------- #

tuned_parameters_svm = {
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 2500, 3000, 3500, 3700],
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'clf__kernel': ['rbf', 'linear']
}

BOW_TFIDF_UNI_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
])

BOW_UNI_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
])

BOW_TFIDF_BI_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
])

BOW_BI_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2))),
    ('fselect', SelectKBest(chi2)),
    ('clf', svm.SVC()),
])

'''
scores = cross_val_score(svm, data, labels, cv=10)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
'''

# ----------------------------------- NAIVE-BAYES ----------------------------------- #

tuned_parameters_naive_bayes = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__alpha': [1, 1e-1, 1e-2]
}

BOW_MultinomialNB = Pipeline([
    ('vect', CountVectorizer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
])

BOW_TFIDF_MultinomialNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', MultinomialNB()),
])

BOW_TFIDF_ComplementNB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', ComplementNB()),
])

BOW_ComplemntNB = Pipeline([
    ('vect', CountVectorizer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', ComplementNB()),
])

'''
scores = cross_val_score(gaussianNB, data, labels, cv=10)
print("Accuracy GaussianNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print(scores)
'''

# ---------------------------------- DECISION TREE ---------------------------------- #

tuned_parameters_decision_tree = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': np.arange(2, 12, 2),
    'clf__min_samples_split': np.arange(1, 10),
    'clf__min_samples_leaf': np.arange(1, 5)
}

BOW_TFIDF_Decision_Tree = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', DecisionTreeClassifier()),
])

BOW_Decision_Tree = Pipeline([
    ('vect', CountVectorizer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', DecisionTreeClassifier()),
])

# ---------------------------------- RANDOM FOREST ---------------------------------- #

tuned_parameters_random_forest = {
    'vect__ngram_range': [(1, 1)],
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__criterion': ['gini', 'entropy'],
    'clf__n_estimators': [100, 300, 500, 750, 800, 1200],
    'clf__min_samples_split': [2, 5, 10],
}

BOW_Random_Forest = Pipeline([
    ('vect', CountVectorizer()),
    ('fselect', SelectKBest(chi2)),
    ('clf', RandomForestClassifier()),
])

# ---------------------------------- LOGISTIC REGRESSION ---------------------------------- #

tuned_parameters_logistic_regression = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

BOW_TFIDF_Logistic_Regression = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', LogisticRegression()),
])

BOW_Logistic_Regression = Pipeline([
    ('vect', stem_vectorizer),
    ('fselect', SelectKBest(chi2)),
    ('clf', LogisticRegression()),
])

# --------------- K-NN ---------------
knn = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', KNeighborsClassifier(n_neighbors=3)),
])

# ---------------------------------- ADABOOST ---------------------------------- #

tuned_parameters_Boost = {
    'vect__ngram_range': [(1, 1)],
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__n_estimators': [100, 500, 1000, 1500],
    'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5]
}

BOW_TFIDF_AdaBoost = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', AdaBoostClassifier()),
])

# ---------------------------------- Gradient ---------------------------------- #

'''
come sopra
'''

BOW_TFIDF_Gradient_Boosting = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', GradientBoostingClassifier()),
])

# ---------------------------------- Bagging ---------------------------------- #

tuned_parameters_Bagging = {
    'vect__ngram_range': [(1, 1)],
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__n_estimators': [10, 30, 50, 100],
}

BOW_TFIDF_Bagging_SVM = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', BaggingClassifier(base_estimator=svm.SVC())),
])

BOW_TFIDF_Bagging_Logistic_Regression = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', BaggingClassifier(base_estimator=LogisticRegression())),
])

BOW_TFIDF_Bagging = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', BaggingClassifier()),
])

models_pipelines = [
    {"model name": "SVM + BOW + TFIDF - UniGram", "model": BOW_TFIDF_UNI_SVM, 'paramiters': tuned_parameters_svm},
    {"model name": "SVM + BOW - UniGram", "model": BOW_UNI_SVM, 'paramiters': tuned_parameters_svm},
    {"model name": "SVM + BOW + TFIDF - BiGram", "model": BOW_TFIDF_BI_SVM, 'paramiters': tuned_parameters_svm},
    {"model name": "SVM + BOW - BiGram", "model": BOW_BI_SVM, 'paramiters': tuned_parameters_svm}
]

'''
    {"model name": "MultinomialNB + BOW + TFIDF", "model": BOW_TFIDF_MultinomialNB, 'paramiters': tuned_parameters_naive_bayes},
    {"model name": "MultinomialNB + BOW", "model": BOW_MultinomialNB, 'paramiters': tuned_parameters_naive_bayes},

    {"model name": "ComplementNB + BOW + TFIDF", "model": BOW_TFIDF_ComplementNB, 'paramiters': tuned_parameters_naive_bayes},
    {"model name": "ComplementNB + BOW", "model": BOW_ComplemntNB, 'paramiters': tuned_parameters_naive_bayes},

    {"model name": "Decision Tree + BOW + TFIDF", "model": BOW_TFIDF_Decision_Tree, 'paramiters': tuned_parameters_decision_tree},
    {"model name": "Decision Tree + BOW", "model": BOW_Decision_Tree, 'paramiters': tuned_parameters_decision_tree},

    {"model name": "Random Forest Classifier + BOW", "model": BOW_Random_Forest, 'paramiters': tuned_parameters_random_forest},

    {"model name": "Logistic Regression + BOW + TFIDF", "model": BOW_TFIDF_Logistic_Regression,'paramiters': tuned_parameters_logistic_regression},
    {"model name": "Logistic Regression + BOW", "model": BOW_Logistic_Regression,'paramiters': tuned_parameters_logistic_regression},


    {"model name": "Adaboost Classifier + BOW + TFIDF", "model": BOW_TFIDF_AdaBoost, 'paramiters': tuned_parameters_Boost},

    {"model name": "Gradient Boosting Classifier + BOW + TFIDF", "model": BOW_TFIDF_Gradient_Boosting, 'paramiters': tuned_parameters_Boost},

    {"model name": "Bagging + SVM + BOW + TFIDF", "model": BOW_TFIDF_Bagging_SVM, 'paramiters': tuned_parameters_Bagging},
    {"model name": "Bagging + Logistic Regression + BOW + TFIDF", "model": BOW_TFIDF_Bagging_Logistic_Regression, 'paramiters': tuned_parameters_Bagging},
    {"model name": "Bagging + BOW + TFIDF", "model": BOW_TFIDF_Bagging, 'paramiters': tuned_parameters_Bagging}
'''

# ------------------------- grid searching best hyper-parameters for each classifier ------------------------- #


for model in models_pipelines:
    name = model['model name']
    pipe = model['model']
    params = model['paramiters']

    model_results = {
        "accuracy mean scores": [],
        "accuracy std scores": [],
        "precision mean scores": [],
        "precision std scores": [],
        "recall mean scores": [],
        "recall std scores": [],
        "f1_score mean scores": [],
        "f1_score std scores": [],
        "best params": []
    }

    clf = GridSearchCV(pipe, params, cv=10, scoring=scoring, refit="accuracy", n_jobs=-1)
    clf.fit(x_train, y_train)
    model_results["best params"].append(clf.best_params_)

    # getting metrics results
    results = clf.cv_results_
    print(results)
    for scorer in scoring:
        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_mean_score = results['mean_test_%s' % scorer][best_index]
        best_std_score = results['std_test_%s' % scorer][best_index]
        results["%s mean scores" % scorer].append(best_mean_score)
        results["%s std scores" % scorer].append(best_std_score)

    print(model_results)
    with open('./MTT_Results/MTT_' + name + '_results', 'w') as fout:
        json.dump(model_results, fout, indent=4)
    # saving best estimator
    # model_name = './tuned_models/' + str(name) + ".pkl"
    # joblib.dump(clf, model_name, compress=1)

"""
# ----------------------- cross test ----------------------- #
for m1 in models_pipelines:
    for m2 in models_pipelines:
        if m1 == m2:
            continue
        print("Testing Models: M1 {} M2 {}".format(m1["model_name"], m2["model_name"]))
        #tTest(m1["model"], m2["model"])"""