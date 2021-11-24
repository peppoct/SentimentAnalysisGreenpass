import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from pre_processing import clening
from statistical_test import t_test
from text_normalization import normalize_text



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

pipelines = [
    {'model name': 'ME_SVM', 'pipeline': BOW_TFIDF_UNI_SVM},
    {'model name': 'ME_Logistic_regression', 'pipeline': BOW_TFIDF_UNI_Logistic_Regression},
    #{'model name': 'ME_MultinomialNB', 'pipeline': model_building.BOW_UNI_MultinomialNB},
    #{'model name': 'ME_ComplementNB', 'pipeline': model_building.BOW_TFIDF_ComplementNB},
    #{'model name': 'ME_Bagging_SVM', 'pipeline': model_building.BOW_TFIDF_UNI_Bagging_SVM},
    #{'model name': 'ME_Bagging_Logistic_Regression', 'pipeline': model_building.BOW_TFIDF_UNI_Bagging_Logistic_Regression}
]

def scoring(pipeline_A, pipeline_B, data, labels, iter):
    results_10CV_A = []
    results_10CV_B = []

    # start iter
    for i in range(1, iter + 1):
        X, y = shuffle(data, labels, random_state=42)
        results_10CV_A.append(np.mean(cross_validate(estimator=pipeline_A,
                                                     X=X,
                                                     y=y,
                                                     cv=10,
                                                     scoring=scorer,
                                                     n_jobs=-1
                                                     )['test_accuracy']))
        results_10CV_B.append(np.mean(cross_validate(estimator=pipeline_B,
                                                     X=X,
                                                     y=y,
                                                     cv=10,
                                                     scoring=scorer,
                                                     n_jobs=-1
                                                     )['test_accuracy']))
    return results_10CV_A, results_10CV_B
    # stop iter

if __name__ == '__main__':
    dataset = pd.read_csv("./dataset/july_to_be_targeted.csv", usecols=['content', 'sentiment'],
                          dtype={'content': 'str', 'sentiment': 'int'})
    dataset = dataset[~dataset['sentiment'].isnull()]
    dataset = dataset.sample(frac=1)
    dataset = clening(dataset)

    data = dataset['content']
    labels = dataset['sentiment']

    # scoring function
    scorer = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, average='micro', labels=labels, zero_division=True),
               'recall': make_scorer(recall_score, average='micro', labels=labels, zero_division=True),
               'f1_score': make_scorer(f1_score, average='micro', labels=labels, zero_division=True)}

    data = normalize_text(data)

    numOfRounds = 10
    sig_level = 0.05

    for model1 in pipelines:
        pipelines.remove(model1)
        for model2 in pipelines:
            if model1 == model2:
                continue
            print("-----------------------------------------------------------------------------------------")
            print("Testing Models:   M1: {}    M2: {}".format(model1["model name"], model2["model name"]))
            accuracy_A, accuracy_B = scoring(model1['pipeline'], model2['pipeline'], data, labels, numOfRounds)
            print(accuracy_A)
            print(accuracy_B)
            print("Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)" % (np.mean(accuracy_A), np.std(accuracy_A) * 2))
            print("Mean Accuracy rate of model M2: %0.4f (+/- %0.4f)" % (np.mean(accuracy_B), np.std(accuracy_B) * 2))
            t, c, res = t_test(accuracy_A, accuracy_B, numOfRounds, sig_level)

            with open('./M_Results/t-test_results', 'w') as fout:
                fout.write("Testing Models:   M1: {}    M2: {}\n".format(model1["model name"], model2["model name"]))
                fout.write("t-statistic: {}\n".format(t))
                fout.write("c-critical: {}\n".format(c))
                fout.write(
                    "Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)\n" % (np.mean(accuracy_A), np.std(accuracy_A) * 2))
                fout.write(
                    "Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)\n" % (np.mean(accuracy_B), np.std(accuracy_B) * 2))
                fout.write("Response: {}\n\n\n".format(res))
                fout.close()
