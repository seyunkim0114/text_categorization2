import numpy as np
import os

import re

import nltk
from nltk.corpus import stopwords
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers


# nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

# list_labeled_train_docs = input("Enter path to a labeled training document: ")
# list_unlabeled_test_docs = input("Enter path to a unlabeled test document: ")

list_labeled_train_docs = "/TC_provided/corpus2_train.labels"
list_unlabeled_test_docs = ""

train_data = pd.DataFrame()
test_data = pd.DataFrame()

MAX_NB_WORDS = 1000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 130
# This is fixed.
EMBEDDING_DIM = 30


def text_preprocess(text):
    """
        text: string to text categorize

        return: modified initial string
    """
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

if list_unlabeled_test_docs == "":
    with open(list_labeled_train_docs, 'r') as f:
        for path in f:
            path = path.rstrip()
            path_to_text = path.split(" ")[0]
            category = path.split(" ")[-1]

            with open(os.path.join(os.path.dirname(list_labeled_train_docs), path_to_text[2:]), 'r') as file:
                line = file.read()
                # print(len(line.split(' ')))
                line = text_preprocess(line)
                temp = pd.DataFrame({"path":[path_to_text], "text":[line], "category":[category], "label":[3]})
                train_data = pd.concat([train_data, temp], ignore_index=True)
                file.close()
            
        f.close()
   
    # # The maximum number of words to be used. (most frequent)
    # MAX_NB_WORDS = 50000
    # # Max number of words in each complaint.
    # MAX_SEQUENCE_LENGTH = 250
    # # This is fixed.
    # EMBEDDING_DIM = 100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train_data['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_train = tokenizer.texts_to_sequences(train_data['text'].values)
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    Y_train = pd.get_dummies(train_data['category'])
    print("1: ", X_train.shape)
    print("2: ", Y_train.shape)
    print("3: ", train_data.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.20, random_state = 42)

else:
    with open(list_labeled_train_docs, 'r') as f:
        for path in f:
            path = path.rstrip()
            path_to_text = path.split(" ")[0]
            category = path.split(" ")[-1]

            with open(os.path.join(os.path.dirname(list_labeled_train_docs), path_to_text[2:]), 'r') as file:
                line = file.read()
                line = text_preprocess(line)
                temp = pd.DataFrame({"path":[path_to_text], "text":[line], "category":[category], "label":[3]})
                train_data = pd.concat([train_data, temp], ignore_index=True)
                file.close()
            
        f.close()

    with open(list_unlabeled_test_docs, 'r') as f:
        for path in f:
            path = path.rstrip()
            path_to_text = path.split(" ")[0]
            category = path.split(" ")[-1]

            with open(os.path.join(os.path.dirname(list_unlabeled_test_docs), path_to_text), 'r') as file:
                line = file.read()
                line = text_preprocess(line)
                temp = pd.DataFrame({"path":[path_to_text], "text":[line], "category":[category], "label":[3]})
                test_data = pd.concat([test_data, temp], ignore_index=True)
                file.close()
        f.close()

    # # The maximum number of words to be used. (most frequent)
    # MAX_NB_WORDS = 50000
    # # Max number of words in each complaint.
    # MAX_SEQUENCE_LENGTH = 250
    # # This is fixed.
    # EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(train_data['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_train = tokenizer.texts_to_sequences(train_data['text'].values)
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    Y_train = pd.get_dummies(train_data['category'])
    print("1: ", X_train.shape)
    print("2: ", Y_train.shape)


    tokenizer_test = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer_test.fit_on_texts(test_data['text'].values)
    word_index_test = tokenizer_test.word_index
    print('Found %s unique tokens in test' % len(word_index_test))

    X_test = tokenizer_test.texts_to_sequences(test_data['text'].values)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    Y_test = pd.get_dummies(test_data['category'])
    print("3: ", X_test.shape)
    print("4: ". Y_test.shape)


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))

model.add(LSTM(EMBEDDING_DIM, dropout=0.2, return_sequences=True, bias_regularizer=regularizers.L2(1e-4)))
# model.add(LSTM(EMBEDDING_DIM, dropout=0.2))
model.add(LSTM(EMBEDDING_DIM, dropout=0.3, bias_regularizer=regularizers.L2(1e-4)))
model.add(Dense(Y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 100
batch_size = 64

history = model.fit(X_train, Y_train, 
                    epochs = epochs,
                    batch_size = batch_size,
                    shuffle=True,
                    validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', 
                                             patience=25,
                                             min_delta=0.0001)]
)


print("MAX NB WORDS: ", MAX_NB_WORDS)
print("EMBEDDING DIM", EMBEDDING_DIM)

accr = model.evaluate(X_test, Y_test)
print(f'Test loss: {accr[0]}, Test acc: {accr[1]}')


# Model evaluation
model_history = pd.DataFrame(history.history)
model_history['epoch'] = history.epoch

fig, ax = plt.subplots(1, figsize=(8,6))
num_epochs = model_history.shape[0]

ax.plot(np.arange(0, num_epochs), model_history["mae"], 
        label="Training MAE")
ax.plot(np.arange(0, num_epochs), model_history["val_mae"], 
        label="Validation MAE")
ax.legend()

plt.tight_layout()
plt.show()

