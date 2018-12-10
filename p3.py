import pandas as pd


def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("cleantextlabels7.csv", names=colnames)
    labels = data.label.tolist()
    tweets = data.tweets.tolist()
    train_tweets = tweets[0: 11001]
    train_labels = labels[0:11001]
    test_tweets = tweets[11001:]
    test_label = labels[11001:]
    return [(train_tweets, train_labels), (test_tweets, test_label)]
if __name__ == '__main__':
    load_data()
    (train_data, train_labels), (test_data, test_labels) = load_data()
    