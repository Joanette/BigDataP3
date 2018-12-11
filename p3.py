import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def convert(arg, dtype):
    arg = tf.convert_to_tensor(
        value=arg,
        dtype=dtype
    )
    return arg

def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("cleantextlabels7.csv", names=colnames)

    labels = data.label.tolist()
    tweets = data.tweets.tolist()

    train_tweets = tweets[0: 11001]
    train_labels = labels[0:11001]

    test_tweets = tweets[11001:]
    test_label = labels[11001:]

    t = Tokenizer()
    t.fit_on_texts(train_tweets)
    train_tweets = t.texts_to_matrix(tweets, mode='count')
    t.fit_on_texts(test_tweets)
    test_tweets = t.texts_to_matrix(tweets, mode='count')

    train_tweets =  keras.preprocessing.sequence.pad_sequences(train_tweets, value= 0, padding='post', maxlen=280)
    test_tweets = keras.preprocessing.sequence.pad_sequences(test_tweets, value=0, padding='post', maxlen=280)

    return [(train_tweets, train_labels),(test_tweets, test_label)]



if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = load_data()
    #print(train_data[0])
    #len(train_data[0]), len(train_data[1])
  
    #define Tokenize with Vocab size
    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(train_data)
    #t.fit_on_texts(train_data)
    #train_tweets = t.texts_to_matrix(tweets, mode='count')
    #t.fit_on_texts(train_data)
    #test_tweets = t.texts_to_matrix(tweets, mode='count')


    model = Sequential()

    model.add(Dense(5, input_shape=(27332, )))
    model.add(Dense(3, activation='softmax'))


    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data,  train_labels, batch_size=64,  epochs=5, steps_per_epoch =1 )
