import numpy as np
import pandas as pd

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

from best_models_paramiters import *
from pre_processing import clening
from statistical_test import t_test
from text_normalization import normalize_text

pipelines = [
    {'model name': 'ME_SVM', 'pipeline': BOW_TFIDF_UNI_SVM},
    {'model name': 'ME_Logistic_regression', 'pipeline': BOW_TFIDF_UNI_Logistic_Regression},
    {'model name': 'ME_MultinomialNB', 'pipeline': BOW_UNI_MultinomialNB},
    {'model name': 'ME_ComplementNB', 'pipeline': BOW_TFIDF_ComplementNB},
    {'model name': 'ME_Bagging_SVM', 'pipeline': BOW_TFIDF_UNI_Bagging_SVM},
    {'model name': 'ME_Bagging_Logistic_Regression', 'pipeline': BOW_TFIDF_UNI_Bagging_Logistic_Regression}
]

def scoring(pipeline_A, pipeline_B, data, labels, iter):
    results_10CV_A = []
    results_10CV_B = []

    # start iter
    for i in range(1, iter + 1):
        X, y = shuffle(data, labels, random_state=i*42)
        results_10CV_A.append(np.mean(cross_val_score(estimator=pipeline_A,
                                                     X=X,
                                                     y=y,
                                                     cv=10,
                                                     n_jobs=-1
                                                     )))
        results_10CV_B.append(np.mean(cross_val_score(estimator=pipeline_B,
                                                     X=X,
                                                     y=y,
                                                     cv=10,
                                                     n_jobs=-1
                                                     )))
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

    # double 'for' to avoid redundant test
    for i in range(0, len(pipelines)-1):
        for j in range(i+1, len(pipelines)):
            print("-----------------------------------------------------------------------------------------")
            print("Testing Models:   M1: {}    M2: {}".format(pipelines[i]["model name"], pipelines[j]["model name"]))
            accuracy_A, accuracy_B = scoring(pipelines[i]['pipeline'], pipelines[j]['pipeline'], data, labels, numOfRounds)
            print("Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)" % (np.mean(accuracy_A), np.std(accuracy_A) * 2))
            print("Mean Accuracy rate of model M2: %0.4f (+/- %0.4f)" % (np.mean(accuracy_B), np.std(accuracy_B) * 2))

            t, c, res = t_test(accuracy_A, accuracy_B, numOfRounds, sig_level)
            print('t statistic: %.3f' % t)
            print(res)
            with open('./Test_Results/t-test_results.txt', 'a') as fout:
                fout.write("***********************************************************************************************\n")
                fout.write("Testing Models:   M1: {}    M2: {}\n".format(pipelines[i]["model name"], pipelines[j]["model name"]))
                fout.write("t-statistic: {}\n".format(t))
                fout.write("c-critical: {}\n".format(c))
                fout.write(
                    "Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)\n" % (np.mean(accuracy_A), np.std(accuracy_A) * 2))
                fout.write(
                    "Mean Accuracy rate of model M1: %0.4f (+/- %0.4f)\n" % (np.mean(accuracy_B), np.std(accuracy_B) * 2))
                fout.write("Response: {}\n".format(res))
                fout.close()
