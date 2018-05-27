# coding: utf-8

# In[1]:
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.utils import np_utils

# In[2]:
# run the commented code to preprocess the data
# import pandas as pd
# import re

# data = pd.read_csv('quotes_all.csv',delimiter=';')

# strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

# def cleanSentences(string):
#     string = string.lower().replace("<br />", " ")
#     return re.sub(strip_special_chars, "", string.lower())

topics = ['death' ,'family', 'freedom' , 'funny', 'life' , 'love', 'happiness', 'success', 'science', 'politics']

# for topic in topics:
#     quotes_data = data[data['Topic'].isin([topic])]

#     with open('%s.txt'%topic,'w+') as quotefile:
#         for quote in quotes_data['Quote']:
#             quotefile.writelines(cleanSentences(quote.lower()) + " ")



# load ascii text and covert to lowercase
filename = "all.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
uniq_words = sorted(list(set(raw_text)))
word_to_int = dict((c, i) for i, c in enumerate(uniq_words))
int_to_word = dict((i, c) for i, c in enumerate(uniq_words))
# summarize the loaded data
n_words = len(raw_text)
n_vocab = len(uniq_words)
print("Total Words in data: %s"% n_words)
print("Vocab (unique words): %s"% n_vocab)


# In[3]:

for topic in topics:

    # In[4]:
    filename = "%s.txt"%topic
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    n_words = len(raw_text)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    seq = []
    next_seq = []
    for i in range(0, n_words - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]

        seq.append([word_to_int[char] for char in seq_in])
        next_seq.append(word_to_int[seq_out])


    # In[5]:

    n_patterns = len(seq)
    print ("Total Patterns: ", n_patterns)
    X = np.reshape(seq, (n_patterns, seq_length, 1))
    X = X / float(n_vocab)
    y = np_utils.to_categorical(next_seq)


    # In[6]:

    def on_epoch_end(epoch,logs):

        start = np.random.randint(0, len(seq)-1)
        pattern = seq[start]

        print("Seed:")
        print("\"", ''.join([int_to_word[value] for value in pattern]), "\"")

        for i in range(1000):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)

            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)

            result = int_to_word[index]
            seq_in = [int_to_word[value] for value in pattern]

            sys.stdout.write(result)

            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print("\nDone.")

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    # In[7]:

    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))

    # filename = "weights-improvement-05-1.8317.hdf5"
    # model.load_weights(filename)

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # define the checkpoint

    filepath="weights-improvement-%s-{epoch:02d}-{loss:.4f}.hdf5"%topic
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [checkpoint, print_callback]
    model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)
