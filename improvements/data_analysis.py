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

    old_set = pd.read_csv('dataset/old_training_set_0.csv')
    old_set = old_set[~old_set['sentiment'].isnull()]
    #print(old_training_set.shape)

    new_labeled_tweets = pd.read_csv('./dataset/pick_' + str(pick) + '.csv')
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


tuned_parameters_naive_bayes = {
    'vect__max_df': (0.65, 0.75, 0.85, 1.0),
    'fselect__k': ['all', 1000, 2000, 3000, 3500, 4000],
    'clf__alpha': [1, 1e-1, 1e-2]
}

BOW_ComplementNB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 1))),
    ('tfidf', TfidfTransformer(smooth_idf=True, use_idf=True)),
    ('fselect', SelectKBest(chi2)),
    ('clf', ComplementNB()),
])

def validate_classifier(mode, peak, x_train, y_train):
    results = {
        "accuracy mean scores": [],
        "accuracy std scores": [],
        "precision mean scores": [],
        "precision std scores": [],
        "recall mean scores": [],
        "recall std scores": [],
        "f1_score mean scores": [],
        "f1_score std scores": [],
        "best params": [],
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='micro', labels=y_train, zero_division=True),
        'recall': make_scorer(recall_score, average='micro', labels=y_train, zero_division=True),
        'f1_score': make_scorer(f1_score, average='micro', labels=y_train, zero_division=True)
    }


    model = BOW_ComplementNB.fit(x_train, y_train)
    results["best params"].append(model.best_params_)

    #model.estimator.named_steps['vect'].get_feature_names()

    model_results = model.cv_results_


    for scorer in scoring:
        best_index = np.nonzero(model_results['rank_test_%s' % scorer] == 1)[0][0]
        best_mean_score = model_results['mean_test_%s' % scorer][best_index]
        best_std_score = model_results['std_test_%s' % scorer][best_index]
        results["%s mean scores" % scorer].append(best_mean_score)
        results["%s std scores" % scorer].append(best_std_score)

    print(model)
    print(results)
    with open('./Monitoring/Complement_'+mode+'_val_result_peak_'+str(peak)+'.txt', 'w') as fout:
        json.dump(results, fout, indent=4)

    # saving best estimator
    model_name = './Monitoring/Complement_'+mode+'_model_peak_' + str(peak) + ".pkl"
    joblib.dump(model, model_name, compress=1)

    return model

def evaluate_classifier(clf, mode, peak, x_test, y_test):

    # Evaluation on test set
    predicted = clf.predict(x_test)  # prediction
    print(clf.classes_, 'Negative', 'Neutral', 'Positive')
    # Extracting statistics and metrics
    report = classification_report(y_test, predicted, labels=clf.classes_, target_names=['Negative', 'Neutral', 'Positive'], digits=4, output_dict=True)
    print(report)
    '''
    with open('./Monitoring/Complement_'+mode+'_test_result_peak_'+str(peak)+'.txt', 'w') as fout:
        json.dump(report, fout, indent=4)
    '''
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('./Monitoring/Complement_'+mode+'_test_result_peak_'+str(peak)+'.csv', index=True)

    cm = confusion_matrix(y_test, predicted, labels=clf.classes_, normalize='true')
    print(clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])

    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.title("Confusion Matrix - peak " + str(peak))

    plt.savefig('./Confusion_Matrix/Confusion_Matrix_Complement_'+mode+'_peak_'+str(peak)+'.png')

def generate_models(peak):
    _, _ = generate_sets(peak)
    incremental_set = pd.read_csv('./dataset/incremental_peak_' + str(peak) + '.csv')
    sliding_set = pd.read_csv('./dataset/sliding_peak_' + str(peak) + '.csv')

    incremental_set = clening(incremental_set)
    sliding_set = clening(sliding_set)

    incremental_data = normalize_text(incremental_set['content'])
    incremental_labels = incremental_set['sentiment']

    sliding_data = normalize_text(sliding_set['content'])
    sliding_labels = sliding_set['sentiment']

    '''
    model = joblib.load('./Monitoring/Complement_incremental_model_peak_6.pkl')
    print(model.best_estimator_[3].n_features_in_)
    '''
    incremental_model = validate_classifier('incremental', peak, incremental_data, incremental_labels)
    sliding_model = validate_classifier('sliding', peak, sliding_data, sliding_labels)
    static_model = joblib.load('./Models/ComplementNB + BOW + TFIDF - UniGram.pkl')

    # test the models on newest tweets
    next_peak = pd.read_csv('./dataset/peak_'+str(peak+1)+'.csv')
    next_peak = next_peak[~next_peak['sentiment'].isnull()]
    print(next_peak.shape)
    next_peak = clening(next_peak)
    next_peak_data = normalize_text(next_peak['content'])
    next_peak_label = next_peak['sentiment']

    evaluate_classifier(incremental_model, 'incremental', peak, next_peak_data, next_peak_label)
    evaluate_classifier(sliding_model, 'sliding', peak, next_peak_data, next_peak_label)
    evaluate_classifier(static_model, 'static', peak, next_peak_data, next_peak_label)

def generate_sets(peak):
    # previous training set
    if (peak-1)==0:
        old_set = pd.read_csv('dataset/old_training_set_'+str(peak-1)+'.csv')
        old_set = old_set[~old_set['sentiment'].isnull()]
        old_set = old_set.sort_values(by=['date'])
        old_incremental = old_set
        old_sliding = old_set
    else:
        old_incremental = pd.read_csv('dataset/incremental_peak_' + str(peak - 1) + '.csv')
        old_incremental = old_incremental.sort_values(by=['date'])
        old_sliding = pd.read_csv('dataset/sliding_peak_' + str(peak - 1) + '.csv')
        old_sliding = old_sliding.sort_values(by=['date'])

    # new labeled tweets
    new_labeled_tweets = pd.read_csv('./dataset/peak_' + str(peak) + '.csv')
    new_labeled_tweets = new_labeled_tweets[~new_labeled_tweets['sentiment'].isnull()]
    new_labeled_tweets = new_labeled_tweets[['id', 'content', 'date', 'sentiment']]

    # incremental model
    incremental_set = pd.concat([old_incremental, new_labeled_tweets], axis=0)
    u.save_dataset(incremental_set, 'incremental_peak_' + str(peak))

    # Sliding model
    old_sliding.drop(old_sliding.head(len(new_labeled_tweets)).index, inplace=True)
    sliding_set = pd.concat([old_sliding, new_labeled_tweets], axis=0)
    u.save_dataset(sliding_set, 'sliding_peak_' + str(peak))
                                        # test static model
    return incremental_set, sliding_set

def generate_peak():
    d = pd.read_csv('./dataset/complete_dataset_cleaned.csv')
    d = d[['id', 'content', 'date']]
    d = d[(d['date']>'11-24-2021') & (d['date']<='12-06-2021')]
    d['content'] = d['content'].str.strip()
    d['content'] = d['content'].replace('\s+', ' ', regex=True)
    d['sentiment'] = ""
    u.save_dataset(d, 'peak_6')
    print(len(d))

if __name__ == '__main__':
    generate_models(6)

    pass