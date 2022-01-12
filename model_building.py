import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from pre_processing import clening
from text_normalization import normalize_text
from sklearn.feature_selection import SelectKBest, chi2

# ------------------------------- Classifier ------------------------------- #
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import matplotlib.pyplot as plt

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
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', LogisticRegression(C=1, max_iter=1500)),
])

# ----------------------------------- NAIVE-BAYES ----------------------------------- #
BOW_TFIDF_UNI_MultinomialNB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k='all')),
    ('clf', MultinomialNB(alpha=1)),
])

BOW_TFIDF_UNI_ComplementNB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=0.65)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3500)),
    ('clf', ComplementNB(alpha=1)),
])

# ---------------------------------- Bagging ---------------------------------- #
BOW_TFIDF_UNI_Bagging_SVM = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=1.0)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k='all')),
    ('clf', BaggingClassifier(base_estimator=svm.SVC(), n_estimators=10)),
])

BOW_TFIDF_UNI_Bagging_Logistic_Regression = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=0.85)),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2, k=3000)),
    ('clf', BaggingClassifier(base_estimator=LogisticRegression(max_iter=1500), n_estimators=100)),
])

pipelines = [
    {'model name': 'BOW_SVM', 'pipeline': BOW_TFIDF_UNI_SVM},
    {'model name': 'BOW_Logistic_regression', 'pipeline': BOW_TFIDF_UNI_Logistic_Regression},
    {'model name': 'BOW_MultinomialNB', 'pipeline': BOW_TFIDF_UNI_MultinomialNB},
    {'model name': 'BOW_ComplementNB', 'pipeline': BOW_TFIDF_UNI_ComplementNB},
    {'model name': 'BOW_Bagging_SVM', 'pipeline': BOW_TFIDF_UNI_Bagging_SVM},
    {'model name': 'BOW_Bagging_Logistic_Regression', 'pipeline': BOW_TFIDF_UNI_Bagging_Logistic_Regression},
]


for pipe in pipelines:
    pr = cross_val_predict(pipe['pipeline'], X=data, y=labels, cv=10, n_jobs=-1)

    report = classification_report(labels, pr, labels=[0, 1, -1], target_names=['Neutral', 'Positive', 'Negative'], digits=4, output_dict=True)
    print(report)
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('./ME_Results/'+pipe['model name']+'_result.csv', index=True)

    cm = confusion_matrix(labels, pr, labels=[0, 1, -1], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])

    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.title("Confusion Matrix - "+pipe['model name'])

    plt.savefig('./Confusion_Matrix/Confusion_Matrix_'+pipe['model name']+'.png')