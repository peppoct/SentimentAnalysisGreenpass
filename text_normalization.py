from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words('italian'))
stemmer = SnowballStemmer('italian')

# stopwords filtering function
def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]

def stem(filtered_sentence):
    return [stemmer.stem(word) for word in filtered_sentence]

def miningfull_word(lemmed_words):
    return [word for word in lemmed_words if len(word) > 2]


def process(t):
    word_tokens = word_tokenize(t)
    word_tokens = remove_stop_words(word_tokens)
    stemmed_words = stem(word_tokens)
    min_words = miningfull_word(stemmed_words)

    processed_tweet = ""
    for word in min_words:
        processed_tweet += word + " "
    return processed_tweet

def normalize_text(dataset):
    normalize_set = []

    for row_index in dataset.index:
        row_field = dataset.loc[row_index]
        normalize_set.append(process(row_field).strip())

    return normalize_set

if __name__ == '__main__':
    pass
