
def save_dataset(tweets, name):
    tweets.to_csv('./dataset/'+name+'.csv', index=False)