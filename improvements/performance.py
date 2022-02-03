import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import ComplementNB


# *********************** LOAD DATA ***************************************************
from sklearn.pipeline import Pipeline

from pre_processing import clening
from text_normalization import normalize_text

dataset = pd.read_csv("../dataset/july_to_be_targeted.csv", usecols=['content', 'sentiment'])
dataset = dataset[~dataset['sentiment'].isnull()]
#dataset = dataset[['content', 'sentimentTest']]
dataset = clening(dataset)
data = dataset['content']
label = dataset['sentiment']
data = normalize_text(data)
# **************************************************************************************


# ****************************** Models *********************************************
# default parameters, no attribute selction
# Complemnt, SVM, Logistic

complement = Pipeline([
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

# *************************************************************************************

pipelines = [
    {'name': 'ComplementNB', 'pipeline': complement},
    {'name': 'LogisticRegression', 'pipeline': logreg},
    {'name': 'SVM', 'pipeline': svm}
]

'''
for pipe in pipelines:
    predicted = cross_val_predict(pipe['pipeline'], X=data, y=label, cv=10, n_jobs=-1)

    report = classification_report(label, predicted, labels=[0, 1, -1], target_names=['Neutral', 'Positive', 'Negative'], digits=4, output_dict=True)
    print(report)
    clsf_report = pd.DataFrame(report).transpose()
    clsf_report.to_csv('./CV_results/'+pipe['name']+'_result.csv', index=True)

    cm = confusion_matrix(label, predicted, labels=[0, 1, -1], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])

    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.title("Confusion Matrix - " + pipe['name'])

    plt.savefig('./CV_results/Confusion_Matrix_' + pipe['name'] + '.png')

'''

if __name__ == '__main__':

    for pipe in pipelines:
        clf = pipe['pipeline']
        clf.fit(data, label)
        # Save to file in the current working directory
        joblib.dump(clf, './models/'+pipe['name']+'.pkl')

        #print(len(clf[0].get_feature_names()))
        # Load from file
        #joblib_model = joblib.load(joblib_file)