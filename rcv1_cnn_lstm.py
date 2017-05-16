'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, LSTM,Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import pickle

import theano
theano.config.openmp = True

BASE_DIR = './data'
GLOVE_DIR = BASE_DIR + '/glove/'
TEXT_DATA_DIR = BASE_DIR + '/rcv1/all/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# Convolution
kernel_size = 5
filters = 64
pool_size = 4


# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = [] # list of label ids

for file_name in sorted(os.listdir(TEXT_DATA_DIR)):
    #file name like: william_wallis=721878newsML.xml.txt
    if False == file_name.endswith(".txt"): continue
    author_name = file_name.split('=')[0]

    if author_name not in labels_index:
        label_id = len(labels_index)
        labels_index[author_name] = label_id

    label_id = labels_index[author_name]
    labels.append(label_id)

    #open file, read each line
    with open(os.path.join(TEXT_DATA_DIR, file_name)) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    text = ''
    for line in lines:
        text += line
    texts.append(text)

print('Found %s texts.' % len(texts))


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Dropout(0.25)(embedded_sequences)
x = Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1)(x)
x = MaxPooling1D(pool_size=pool_size)(x)
x = LSTM(128)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

stopCondition = False
prev_acc = 0
while not stopCondition:
    model.fit(x_train, y_train,
              batch_size=128,
              epochs=2,
              validation_data=(x_val, y_val))

    score, acc = model.evaluate(x_val, y_val,
                                batch_size=128)
    print('Test score:', score)
    print('Test accuracy:', acc)
    relativeErr = abs(acc - prev_acc)/prev_acc
    print('relative error:', relativeErr)
    stopCondition = (relativeErr <= 0.01)
    prev_acc = acc

# serialize model to JSON
model_json = model.to_json()
with open("rcv1.cnn_lstm.model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
pickle.dump(model.get_weights(), open("rcv1.cnn_lstm.pickle", "wb"))

print("Saved model to disk")