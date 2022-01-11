import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from pre_processing import clening
from text_normalization import normalize_text

dataset = pd.read_csv("./dataset/peak_1.csv", usecols=['content', 'sentiment'])
dataset = dataset[~dataset['sentiment'].isnull()]
#dataset = dataset[['content', 'sentimentTest']]
dataset = clening(dataset)
data = dataset['content']
label = dataset['sentiment']
data = normalize_text(data)


def evaluate_classifier(clf, name):
    print(name)
    # Evaluation on test set
    predicted = clf.predict(data)  # prediction

    # Extracting statistics and metrics
    print(classification_report(label, predicted, labels=clf.classes_, target_names=['Negative', 'Neutral', 'Positive'], digits=4))

    cm = confusion_matrix(label, predicted, labels=clf.classes_, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Neutral', 'Positive'])

    disp.plot(cmap=plt.cm.Blues, values_format='g')
    plt.title("Confusion Matrix: " + name)
    #disp.ax_.set_title('Confusion Matrix ' + name)
    plt.savefig('./Confusion_Matrix/Confusion_Matrix_'+name+'.png')


if __name__ == '__main__':
    '''
    bagging = BOW_TFIDF_UNI_Bagging_Logistic_Regression
    complement = BOW_ComplementNB
    multinomial = BOW_TFIDF_BI_MultinomialNB
    '''
    bagging = joblib.load('./Models/Bagging + Logistic Regression + BOW + TFIDF - UniGram.pkl')
    complement = joblib.load('./Models/ComplemnetNB + BOW - UniGram.pkl')
    multinomial = joblib.load('./Models/MultinomialNB + BOW + TFIDF - BiGram.pkl')
    evaluate_classifier(bagging, 'Bagging_Log_Reg')
    evaluate_classifier(complement, 'ComplementNB')
    evaluate_classifier(multinomial, 'MultinomiaNB')





