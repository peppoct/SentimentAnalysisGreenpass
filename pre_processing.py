import re
import pandas as pd
import utils as u
import string

def format(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Remove punctuation
    tweet = tweet.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter : funnnnny --> funny
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)
    # Remove - & '
    tweet = re.sub(r'(-|\')', '', tweet)
    # Remove digit
    tweet = re.sub(r'\d+', '', tweet)
    # Remove # but mantein hashtag text
    tweet = tweet.replace('#', '')
    return tweet

def remove_punctuation(row_field):
    # removing punctuation signs
    for punct_sign in string.punctuation:
        row_field = row_field.replace(punct_sign, " ")
        row_field = row_field.replace("’", " ")
        row_field = row_field.replace("”", " ")
        tweet = row_field.replace("\n", " ")

    return tweet

def remove_from_string(tweet):
    # Replaces URLs with
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    #tweet = re.sub(r'#(\S+)', '', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)

    return tweet


#EMO_POS / EMO_NEG per ora li tolgo perchè per i topoic non mi interessano le reazioni
def remove_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', '', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', '', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', '', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', '', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', '', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', '', tweet)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    return tweet

def clean(tweet):
    tweet = remove_from_string(tweet)
    tweet = remove_emojis(tweet)
    tweet = format(tweet)
    tweet = remove_punctuation(tweet)
    # Remove spaces
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet


def clening(dataset):
    for row_index, tweet in dataset.iterrows():

        processed_tweet = clean(tweet.content)
        if (len(processed_tweet) > 0):
            dataset.loc[row_index, "content"] = processed_tweet
        else:
            dataset.drop(index=row_index, inplace=True)

    return dataset


if __name__ == '__main__':
    pass