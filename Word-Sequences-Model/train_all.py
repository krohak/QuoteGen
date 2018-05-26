# coding: utf-8

# In[1]:
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.utils import np_utils

# In[2]:

# import pandas as pd
# import re

# data = pd.read_csv('quotes_all.csv',delimiter=';')

# strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

# def cleanSentences(string):
#     string = string.lower().replace("<br />", " ")
#     return re.sub(strip_special_chars, "", string.lower())

topics = [#'death' , 
    'family', 'freedom' , 'funny', 'life' , 'love', 'happiness', 'success', 'science', 'politics']

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
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: %s"% n_chars)
print("Total Vocab: %s"% n_vocab)


# In[3]:

for topic in topics:
    
    # In[4]:
    filename = "%s.txt"%topic
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    n_chars = len(raw_text)
    
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])


    # In[5]:

    n_patterns = len(dataX)
    print ("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    # define the LSTM model


    # In[6]:

    def on_epoch_end(epoch,logs):
        # pick a random seed
        start = numpy.random.randint(0, len(dataX)-1)
        pattern = dataX[start]
        print("Seed:")
        print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(1000):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = int_to_char[index]
            seq_in = [int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        print("\nDone.")


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

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    callbacks_list = [checkpoint, print_callback]
    # fit the model
    model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list)

