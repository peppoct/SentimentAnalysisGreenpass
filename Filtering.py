import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

def removeStopWords(stopWords, word_list):
    filtered_words = [word for  word in word_list if word not in stopWords]
    return  filtered_words


def lemmatize():
    wordnet_lemmatizer = WordNetLemmatizer()
    usual_stemmer = SnowballStemmer('italian')

# -------------- load data ---------- #
dataset = pd.read_csv("C:\Users\39351\Desktop\AIDE\Data Mining and ML\DataMiningPROJECT\training_set_july.csv")
