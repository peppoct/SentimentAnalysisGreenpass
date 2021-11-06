import math
import pandas as pd
import numpy as np

from scipy import stats
import model_building
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle

from pre_processing import clening
from text_normalization import normalize_text


def t_test(pipeline_A, pipeline_B, data, labels, iter):
    results_A = []
    results_B = []
    model_A = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }
    model_B = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": []
    }

    for i in range(1, iter+1):
        X, y = shuffle(data, labels, random_state=i*320391)
        if (i%10 == 0):
            print("Iteration #{}".format(i))

        results_A.append(cross_validate(estimator=pipeline_A,
                                        X=X,
                                        y=y,
                                        cv=10,
                                        scoring=scoring,
                                        n_jobs=-1
                                        ))
        results_B.append(cross_validate(estimator=pipeline_B,
                                        X=X,
                                        y=y,
                                        cv=10,
                                        scoring=scoring,
                                        n_jobs=-1
                                        ))

    for result in results_A:
        model_A["Accuracy"].append(np.mean(result['test_accuracy']))
        model_A["Precision"].append(np.mean(result['test_precision']))
        model_A["Recall"].append(np.mean(result['test_recall']))
        model_A["F1 Score"].append(np.mean(result['test_f1_score']))

    for result in results_B:
        model_B["Accuracy"].append(np.mean(result['test_accuracy']))
        model_B["Precision"].append(np.mean(result['test_precision']))
        model_B["Recall"].append(np.mean(result['test_recall']))
        model_B["F1 Score"].append(np.mean(result['test_f1_score']))

    mean_A = np.mean(model_A['Accuracy'])
    mean_B = np.mean(model_B['Accuracy'])
    delta_means = mean_A - mean_B
    delta_sum = 0

    #compute the variance of the difference between the two models
    for i in range(0,iter):
        delta_i = model_A["Accuracy"][i] - model_B["Accuracy"][i]
        delta_sum += (delta_i - delta_means)*(delta_i - delta_means)
    variance = delta_sum/iter

    #t-statistic
    t_statistic = delta_means/math.sqrt(variance/iter)
    print('t statistic: %.3f' % t_statistic)

    # degrees of freedom
    df = len(model_A) + len(model_B) - 2.0

    p_val = stats.t.cdf(t_statistic, df=df)
    print('p value: %.3f' % p_val)

    if t_statistic > 0.025 or t_statistic < -0.025: #if t_statistic
        response = 'We can reject the null-hypothesis that both models perform equally well on this dataset. We may conclude that the two algorithms are significantly different.'
        print(response)
    else:
        response = 'We cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different.'
        print(response)

    return t_statistic, p_val, response

dataset = pd.read_csv("./dataset/800a.csv", usecols=['content', 'sentiment'],
                      dtype={'content': 'str', 'sentiment': 'int'})
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
}

pipelines = [
    {'model name': 'ME_SVM', 'pipeline': model_building.BOW_TFIDF_UNI_SVM},
    {'model name': 'ME_Logistic_regression', 'pipeline': model_building.BOW_TFIDF_UNI_Logistic_Regression},
    {'model name': 'ME_MultinomialNB', 'pipeline': model_building.BOW_UNI_MultinomialNB},
    {'model name': 'ME_ComplementNB', 'pipeline': model_building.BOW_TFIDF_ComplementNB},
    {'model name': 'ME_Bagging_SVM', 'pipeline': model_building.BOW_TFIDF_UNI_Bagging_SVM},
    {'model name': 'ME_Bagging_Logistic_Regression', 'pipeline': model_building.BOW_TFIDF_UNI_Bagging_Logistic_Regression}
]


for model1 in pipelines:
    for model2 in pipelines:
        if model1 == model2:
            continue
        print("-----------------------------------------------------------------------------------------")
        print("Testing Models:   M1: {}    M2: {}".format(model1["model name"], model2["model name"]))
        t_stat, p_v, resp = t_test(model1['pipeline'], model2['pipeline'], data, labels, 10)
        with open('./MTT_Results/t-test_results', 'w') as fout:
            fout.write("Testing Models:   M1: {}    M2: {}".format(model1["model name"], model2["model name"]))
            fout.write("\n")
            fout.write("t-statistic: {}\n".format(t_stat))
            fout.write("p-val: {}\n".format(p_v))
            fout.write("Response: {}".format(resp))
            fout.write("\n\n\n")
            fout.close()











