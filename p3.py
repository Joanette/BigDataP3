import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, MaxPool1D, Embedding
import numpy as np


def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("/home/joanette_rosario/BigDataP3/text10.csv", names=colnames)

    labels = data.label.tolist()
    tweets = data.tweets.tolist()

    train_tweets = tweets[0: 11001]
    train_labels = labels[0:11001]

    test_tweets = tweets[11001:]
    test_label = labels[11001:]

    t = Tokenizer(num_words=100000)
    t.fit_on_texts(train_tweets)
    train_tweets = t.texts_to_sequences(train_tweets)
    t.fit_on_texts(test_tweets)
    test_tweets = t.texts_to_sequences(test_tweets)

    train_tweets =keras.preprocessing.sequence.pad_sequences(train_tweets, value=0, padding='post', maxlen=280)
    test_tweets = keras.preprocessing.sequence.pad_sequences(test_tweets, value=0, padding='post', maxlen=280)

    return [(train_tweets, train_labels),(test_tweets, test_label)]



if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = load_data()
    model = Sequential()
    model.add(Embedding(100000, 16))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dense(512, activation=tf.nn.relu))
    model.add(Dense(16, activation=tf.nn.relu))
    model.add(Dense(3, activation=tf.nn.softmax))
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data,  train_labels, batch_size=512,  epochs=20)

    #evaluate the model
    results = model.evaluate(test_data, test_labels)
    print results