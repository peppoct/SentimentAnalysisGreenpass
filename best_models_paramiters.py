from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import BaggingClassifier

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