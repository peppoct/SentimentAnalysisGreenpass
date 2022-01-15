import joblib
import pandas as pd

import utils
from pre_processing import clening
from text_normalization import normalize_text
from matplotlib import pyplot as plt
import numpy as np




def evaluate_classifier(clf, name):
    dataset = pd.read_csv("./dataset/july.csv")
    dataset1 = pd.read_csv("./dataset/2picco.csv", low_memory=False)
    dataset1 = pd.read_csv("./dataset/postpeak6_ordered.csv")
    data = dataset['content', 'sentiment']
    dataset = dataset[~dataset['content'].isnull()]
    dataset = dataset[~dataset['sentiment'].isnull()]
    dataset = clening(dataset)
    data = dataset['content']
    data = normalize_text(data)


    print(name)
    utils.save_dataset(dataset, 'dataset')
    # Evaluation on test set
    predicted = clf.predict(data)  # prediction
    print(clf.classes_, 'Negative', 'Neutral', 'Positive')

    print(predicted)
    array = pd.Series(predicted)
    utils.save_dataset(array, 'array')
    print(array)


    negative = 0
    neutral = 0
    positive = 0

    for value in predicted:
        if value == -1:     # negative tweets
            negative += 1

        if value == 0:      # neutral tweets
            neutral += 1

        if value == 1:      # positive tweets
            positive += 1

    lengths = [positive, neutral, negative]
    labels = ['Positive', 'Neutral', 'Negative']


def plot():
    july = [27963, 26645, 46447]
    august = [26303, 29172, 49528]
    september = [23668, 31389, 53674]
    october = [14234, 25876, 39094]
    november = [30602, 41884, 75420]
    december = [4060, 4981, 8312]

    labels = ['Positive', 'Negative', 'Neutral']

    fig = plt.figure(figsize=(10, 7))
    colors = ['#4A6A91', '#3270BD', '#B3CBE8']
    plt.pie(july, labels=labels, autopct='%.2f', colors=colors, textprops={'fontsize': 12})
    plt.title(label='July')
    plt.savefig('./dataset/July.svg', format="svg")
    plt.show()


if __name__ == '__main__':
    complement = joblib.load('./Models/ComplementNB + BOW + TFIDF - UniGram.pkl')
    evaluate_classifier(complement, 'ComplementNB')
    #plot()




