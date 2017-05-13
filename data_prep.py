from __future__ import print_function

'''
build rcv1 into numpy format dataset
'''

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


if sys.argv != 3:
    print('need: glove_file data_loc')

GLOVE_LOC = sys.argv[1] # from command line parameter
TEXT_DATA_DIR = sys.argv[2] # from command line parameter
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

# [1] build index mapping words in the embeddings set to their embedding vector
print('Indexing word vectors.')

'''
embeddings_index = {}
f = open(GLOVE_LOC)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
'''

# [2] prepare text samples and labels
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

# [3] vectorize text into a 2D tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

np.save('rcv1.data', data)
np.save('rcv1.label', labels)

# [4] split the data into a training set and a validation set
#indices = np.arange(data.shape[0])
#np.random.shuffle(indices)
#data = data[indices]
#labels = labels[indices]
#num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

#x_train = data[:-num_validation_samples]
#y_train = labels[:-num_validation_samples]
#x_val = data[-num_validation_samples:]
#y_val = labels[-num_validation_samples:]

