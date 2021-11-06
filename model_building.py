import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from pre_processing import clening
from text_normalization import normalize_text
from sklearn.feature_selection import SelectKBest, chi2

# ------------------------------- Classifier ------------------------------- #
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm



dataset = pd.read_csv("./dataset/july_to_be_targeted.csv", usecols=['content', 'sentiment'],
                      dtype={'content': 'str', 'sentiment': 'int'})
dataset = dataset[~dataset['sentiment'].isnull()]
dataset = dataset.sample(frac=1)
dataset = clening(dataset)

data = dataset['content']
labels = dataset['sentiment']

data = normalize_text(data)

# ------------------------------------- SVM ------------------------------------- #

BOW_TFIDF_UNI_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k='all')),
    ('clf', svm.SVC(C=10, gamma=1, kernel='rbf')),
])

# ---------------------------------- LOGISTIC REGRESSION ---------------------------------- #

BOW_TFIDF_UNI_Logistic_Regression = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k='all')),
    ('clf', LogisticRegression(C=10, max_iter=1500)),
])


# ----------------------------------- NAIVE-BAYES ----------------------------------- #

BOW_UNI_MultinomialNB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k='all')),
    ('clf', MultinomialNB(alpha=0.1)),
])

BOW_TFIDF_ComplementNB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k='all')),
    ('clf', ComplementNB(alpha=1)),
])

# ---------------------------------- RANDOM FOREST ---------------------------------- #

BOW_UNI_Random_Forest = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=0.75)),
    ('fselect', SelectKBest(chi2, k=4000)),
    ('clf', RandomForestClassifier(min_samples_split=10, n_estimators=100, criterion='entropy')),
])


# ---------------------------------- Gradient ---------------------------------- #

BOW_TFIDF_UNI_Gradient_Boosting = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=1.0)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=4000)),
    ('clf', GradientBoostingClassifier(n_estimators=500, learning_rate=0.2)),
])

# ---------------------------------- Bagging ---------------------------------- #

BOW_TFIDF_UNI_Bagging_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=2000)),
    ('clf', BaggingClassifier(base_estimator=svm.SVC(C=1, kernel='rbf', gamma=1), n_estimators=10)),
])

BOW_TFIDF_UNI_Bagging_Logistic_Regression = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=1.0)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=4000)),
    ('clf', BaggingClassifier(base_estimator=LogisticRegression(max_iter=1500), n_estimators=50)),
])

pipelines = [
    {'model name': 'ME_SVM', 'pipeline': BOW_TFIDF_UNI_SVM},
    {'model name': 'ME_Logistic_regression', 'pipeline': BOW_TFIDF_UNI_Logistic_Regression},
    {'model name': 'ME_Random_Forest', 'pipeline': BOW_UNI_Random_Forest},
    {'model name': 'ME_MultinomialNB', 'pipeline': BOW_UNI_MultinomialNB},
    {'model name': 'ME_ComplementNB', 'pipeline': BOW_TFIDF_ComplementNB},
    {'model name': 'ME_Gradient_Boosting', 'pipeline': BOW_TFIDF_UNI_Gradient_Boosting},
    {'model name': 'ME_Bagging_SVM', 'pipeline': BOW_TFIDF_UNI_Bagging_SVM},
    {'model name': 'ME_Bagging_Logistic_Regression', 'pipeline': BOW_TFIDF_UNI_Bagging_Logistic_Regression},
]

for pipe in pipelines:
    pr = cross_val_predict(pipe['pipeline'], X=data, y=labels, cv=10, n_jobs=-1)
    print(pipe['model name'])
    print(classification_report(labels, pr, labels=[0, 1, -1], target_names=['Neutral', 'Positive', 'Negative'], digits=4))