
def save_dataset(tweets, name):
    tweets.to_csv('./dataset/data/'+name+'.csv', index=False)