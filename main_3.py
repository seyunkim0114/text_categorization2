import numpy as np
import os

import re

import nltk
from nltk.corpus import stopwords
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
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

list_labeled_train_docs = "/TC_provided/corpus3_train.labels"
list_unlabeled_test_docs = ""

train_data = pd.DataFrame()
test_data = pd.DataFrame()

train_label_map = {}
MAX_NB_WORDS = 1000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 200
# This is fixed.
EMBEDDING_DIM = 60


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
        cat_count = 0
        for path in f:
            path = path.rstrip()
            path_to_text = path.split(" ")[0]
            category = path.split(" ")[-1]
            if not category in train_label_map.keys():
                train_label_map[category] = str(cat_count)
                cat_count += 1
            with open(os.path.join(os.path.dirname(list_labeled_train_docs), path_to_text[2:]), 'r') as file:
                line = file.read()
                # print(len(line.split(' ')))
                line = text_preprocess(line)
                temp = pd.DataFrame({"path":[path_to_text], "text":[line], "category":[category], "label":[int(train_label_map[category])]})
                train_data = pd.concat([train_data, temp], ignore_index=True)
                file.close()
            
        f.close()
    # print(train_label_map)
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
    # print(train_data['category'])
    X_train = tokenizer.texts_to_sequences(train_data['text'].values)
    # print(len(X_train))
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    Y_train = pd.get_dummies(train_data['category'])
    print("1: ", X_train.shape)
    print("2: ", Y_train.shape)
    print("3: ", train_data.shape)
    
    # print(Y_train)
    # print(Y_train.columns)
    # print(Y_train.idxmax(axis=1))
    # print(set(Y_train.idxmax(axis=1)))


    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.20, random_state = 42)
    # print(Y_train)
    # print(Y_test)
    dum_to_classes = Y_train.columns
    # max_X_train = max([max(X) for X in X_train])
    # max_X_test = max([max(X) for X in X_test])
    # X_train = X_train / max_X_train
    # X_test = X_test / max_X_test
    
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

# model.add(LSTM(EMBEDDING_DIM, return_sequences=True, bias_regularizer=regularizers.L2(1e-4)))
# model.add(LSTM(EMBEDDING_DIM, return_sequences=True, bias_regularizer=regularizers.L2(1e-4)))
model.add(LSTM(EMBEDDING_DIM, dropout=0.3, bias_regularizer=regularizers.L2(1e-4)))
model.add(Dense(Y_train.shape[1], activation='softmax'))

lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps = 10000,
    decay_rate = 0.9
)
optim = keras.optimizers.Adam(learning_rate = lr_scheduler)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

epochs = 20
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
print("MAX_SEQUENCE_LENGTH", MAX_SEQUENCE_LENGTH)

accr = model.evaluate(X_test, Y_test)
print(f'Test loss: {accr[0]}, Test acc: {accr[1]}')


# Model evaluation
model_history = pd.DataFrame(history.history)
model_history['epoch'] = history.epoch

fig, ax = plt.subplots(1, figsize=(8,6))
num_epochs = model_history.shape[0]

ax.plot(np.arange(0, num_epochs), model_history["loss"], 
        label="Training loss")
ax.plot(np.arange(0, num_epochs), model_history["val_loss"], 
        label="Validation loss")
ax.legend()

plt.tight_layout()
plt.savefig("temp_corpus3.png")


# Save as output file 
# y_proba = model.predict(X_test)
# y_classes = y_proba.argmax(axis=-1)
# with open("temp_corpus3_output_results", 'w') as f:
#     for i in range(len(y_proba)):
#         f.write(f'{test_data.iloc[i]["path"} {dum_to_classes][y_classes[i]]}]')
