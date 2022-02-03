import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import utils as u

#tweet fields
# id
# date
# content
# likeCount
# lang
from pre_processing import clening
from text_normalization import normalize_text


def filter_by_lang(tweets):
    return tweets[tweets['lang'] == 'it']

def date_format(tweets):
    return pd.to_datetime(tweets["date"]).dt.strftime("%m-%d-%Y")


def generate_training_sets(pick):

    old_set = pd.read_csv('../dataset/old_training_set_0.csv')
    old_set = old_set[~old_set['sentiment'].isnull()]
    #print(old_training_set.shape)

    new_labeled_tweets = pd.read_csv('../dataset/pick_' + str(pick) + '.csv')
    new_labeled_tweets = new_labeled_tweets[~new_labeled_tweets['sentiment'].isnull()]
    new_labeled_tweets = new_labeled_tweets[['id', 'content', 'date', 'sentiment']]
    #print(new_labeled_tweets.shape)

    # incremental model
    incremental_set = pd.concat([old_set, new_labeled_tweets], axis=0)
    #print(incremental_training.shape)
    u.save_dataset(incremental_set, 'incremental_pick_' + str(pick))

    # Sliding model
    old_set.drop(old_set.head(360).index, inplace=True)
    sliding_set = pd.concat([old_set, new_labeled_tweets], axis=0)
    #print(sliding_training.shape)
    u.save_dataset(sliding_set, 'sliding_pick_' + str(pick))
    return incremental_set, sliding_set



complementNB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1), max_df=1.0)),
    ('tfidf', TfidfTransformer()),
    ('clf', ComplementNB()),
])

logreg = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=1.0)),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression()),
])

svm = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,1), max_df=1.0)),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.SVC()),
])

pipelines = [
    {'name': 'ComplementNB', 'pipeline': complementNB},
    {'name': 'LogisticRegression', 'pipeline': logreg},
    {'name': 'SVM', 'pipeline': svm}
]


def train_classifier(clf, name, mode, peak, x_train, y_train):

    model = clf.fit(x_train, y_train)

    # saving best estimator
    model_name = './models/'+name+'_'+mode+'_model_peak_' + str(peak) + ".pkl"
    joblib.dump(model, model_name, compress=1)

    print(len(clf[0].get_feature_names()))


def evaluate_classifier(clf, name, mode, peak, x_test, y_test):

    # Evaluation on test set
    predicted = clf.predict(x_test)  # prediction
    print(clf.classes_, 'Negative', 'Neutral', 'Positive')
    # Extracting statistics and metrics
    report = classification_report(y_test, predicted, labels=clf.classes_, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
    print(report)

    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('./Monitoring_Results/'+name+'_'+mode+'_test_result_peak_'+str(peak)+'.csv', index=True)


def generate_models(peak, pipe):

    incremental_set = pd.read_csv('../dataset/incremental_peak_' + str(peak) + '.csv')
    sliding_set = pd.read_csv('../dataset/sliding_peak_' + str(peak) + '.csv')


    incremental_set = clening(incremental_set)
    sliding_set = clening(sliding_set)

    incremental_data = normalize_text(incremental_set['content'])
    incremental_labels = incremental_set['sentiment']

    sliding_data = normalize_text(sliding_set['content'])
    sliding_labels = sliding_set['sentiment']


    train_classifier(pipe['pipeline'], pipe['name'], 'incremental', peak, incremental_data, incremental_labels)
    train_classifier(pipe['pipeline'], pipe['name'], 'sliding', peak, sliding_data, sliding_labels)

    incremental_model = joblib.load('models/' + pipe['name'] + '_incremental_model_peak_' + str(peak) + '.pkl')
    sliding_model = joblib.load('models/' + pipe['name'] + '_sliding_model_peak_' + str(peak) + '.pkl')
    static_model = joblib.load('models/' + pipe['name'] + '.pkl')

    # test the models on newest tweets
    next_peak = pd.read_csv('../dataset/peak_'+str(peak+1)+'.csv')
    next_peak = next_peak[~next_peak['sentiment'].isnull()]
    print(next_peak.shape)
    next_peak = clening(next_peak)
    next_peak_data = normalize_text(next_peak['content'])
    next_peak_label = next_peak['sentiment']

    evaluate_classifier(incremental_model, pipe['name'], 'incremental', peak, next_peak_data, next_peak_label)
    evaluate_classifier(sliding_model, pipe['name'], 'sliding', peak, next_peak_data, next_peak_label)
    evaluate_classifier(static_model, pipe['name'], 'static', peak, next_peak_data, next_peak_label)

if __name__ == '__main__':
    for pipe in pipelines:
        generate_models(6, pipe)

    pass