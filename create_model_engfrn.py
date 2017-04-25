from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop,SGD
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

text_eng = open('../proj2-master/data/eng.txt').read().lower()
print('english corpus length:', len(text_eng))
chars_eng = sorted(list(set(text_eng)))
print('total chars english :', len(chars_eng))
char_indices_eng = dict((c, i) for i, c in enumerate(chars_eng))
indices_char_eng = dict((i, c) for i, c in enumerate(chars_eng))


text_frn = open('../proj2-master/data/frn.txt').read().lower()
print('french corpus length:', len(text_frn))
chars_frn = sorted(list(set(text_frn)))

print('total chars french :', len(chars_frn))

char_indices_frn = dict((c, i) for i, c in enumerate(chars_frn))
indices_char_frn = dict((i, c) for i, c in enumerate(chars_frn))

chars = set(chars_eng)
chars = sorted(list(chars.union(set(chars_frn))))
print('total chars :', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

text_eng_train = text_eng[:int(0.8*len(text_eng))]
text_eng_test = text_eng[int(0.8*len(text_eng)):]
text_frn_train = text_frn[:int(0.8*len(text_frn))]
text_frn_test = text_frn[int(0.8*len(text_frn)):]

def get_sentences(text, maxlen, step):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    #print('nb sequences:', len(sentences_eng))
    return sentences, next_chars

maxlen = 40
step = 1
sentences_eng, next_chars_eng = get_sentences(text_eng_train, maxlen, step)
sentences_frn, next_chars_frn = get_sentences(text_frn_train, maxlen, step)

def get_sentences_test(text, maxlen, step):
    sentences = []
    next_string = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_string.append(text[i + maxlen:i + maxlen +5])
    return sentences, next_string

maxlen = 40
step = 20
sentences_eng_test, next_string_eng_test = get_sentences_test(text_eng_test, maxlen, step)
sentences_frn_test, next_string_frn_test = get_sentences_test(text_frn_test, maxlen, step)
sentences_eng_test = sentences_eng_test[:100]
next_string_eng_test = next_string_eng_test[:100]
sentences_frn_test = sentences_frn_test[:100]
next_string_frn_test = next_string_frn_test[:100]

def get_vectors(sentences, chars, char_indices, next_chars):
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X,y

X_eng, y_eng = get_vectors(sentences_eng, chars, char_indices, next_chars_eng)
X_frn, y_frn = get_vectors(sentences_frn, chars, char_indices, next_chars_frn)

def build_model(chars, maxlen, X, y,name):
    model = Sequential()
    #model.add(LSTM(256, input_shape=(maxlen, len(chars)), return_sequences=True))
    #model.add(LSTM(128))
    #model.add(LSTM(10))
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    #model.add(Activation('tanh'))
    optimizer = RMSprop(lr=0.01)
    #SGD(lr=0.01, clipnorm=1.)
    #RMSprop(lr=0.01)
    #'categorical_crossentropy'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X, y, batch_size=128, epochs=5)
    model.save(name)
    return model

model_eng = build_model(chars, maxlen, X_eng, y_eng,'eng.h5')
model_frn = build_model(chars, maxlen, X_frn, y_frn,'frn.h5')





