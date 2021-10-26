
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import metrics

from pre_processing import clening
from text_normalization import normalize_text

dataset = pd.read_csv("./dataset/july_to_be_targeted.csv")
dataset = dataset[~dataset['sentiment'].isnull()]
dataset = dataset[['content', 'sentiment']]
dataset = clening(dataset)
data = dataset['content']
label = dataset['sentiment']
data = normalize_text(data)

print("dataset len: " + str(len(dataset)))
print("class 1 len: " + str(len(dataset[label == "1"])))
print("class 0 len: " + str(len(dataset[label == "0"])))
print("class -1 len: " + str(len(dataset[label == "-1"])) + '\n')

# splitting Training and Test set
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

# counting the word occurrences
count_vect = CountVectorizer()
# count_vect = CountVectorizer(stop_words=stopwords,analyzer=stemming,min_df=2)
X_train_counts = count_vect.fit_transform(X_train)
print("List of the extracted tokens")
print(count_vect.get_feature_names())

print("Description of the word occurrences data structures:")
print(type(X_train_counts))
print("(Documents, Tokens)")
print(X_train_counts.shape)
print("Word occurrences of the first document:")
print(X_train_counts[0])
# extracted tokens
# print(count_vect.get_feature_names())

# Text representation supervised stage on training set
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)  # include calculation of TFs (frequencies)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("Values of features extracted from the first document:")
print(X_train_tfidf[0])


# TF-IDF extraction on test set
X_test_counts = count_vect.transform(X_test)  # tokenization and word counting
X_test_tfidf = tfidf_transformer.transform(X_test_counts)  # feature extraction


def evaluate_classifier(clf):
    clf.fit(X_train_tfidf, y_train)
    # Evaluation on test set
    predicted = clf.predict(X_test_tfidf)  # prediction
    # Extracting statistics and metrics
    accuracy = np.mean(predicted == y_test)  # accuracy extreaction
    print('accuracy : ' + str(accuracy))

    f_score = f1_score(y_test, predicted, average='macro')
    print('f_score : ' + str(f_score) + '\n')

    disp = plot_confusion_matrix(clf, X_test_tfidf, y_test, cmap=plt.cm.Blues, normalize='true')
    disp.ax_.set_title('Confusion Matrix')
    plt.show()


# --------------- DECISION TREE ---------------
clf = tree.DecisionTreeClassifier()
print('Decision Tree:')
evaluate_classifier(clf)

# --------------- RANDOM FOREST ---------------
clf2 = RandomForestClassifier()
print('Random Forest:')
evaluate_classifier(clf2)

# --------------- LOGISTIC REGRESSION ---------------
clf3 = LogisticRegression()
print('Logistic Regression:')
evaluate_classifier(clf3)

# --------------- SVC ---------------
clf4 = svm.LinearSVC()
print('SVM:')
evaluate_classifier(clf4)

# --------------- NAIVE-BAYES ---------------
clf2 = MultinomialNB()
print('Multinomial NB:')
evaluate_classifier(clf2)

# --------------- K-NN ---------------
k_neighbor = 5
clf5 = KNeighborsClassifier(k_neighbor)
print('k-NN (k = ' + str(k_neighbor) + ') :')
evaluate_classifier(clf5)

# --------------- ADABOOST ---------------
clf6 = AdaBoostClassifier()
print('Adaboost:')
evaluate_classifier(clf6)

# --------------- GBC ---------------
clf7 = GradientBoostingClassifier()
print('Gradient Boosting:')
evaluate_classifier(clf7)

