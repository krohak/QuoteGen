
# coding: utf-8

# In[1]:

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import random
import sys


# In[2]:

np.random.seed()

ids = np.load('quote_matrix10.npy')
#ids = ids[:1000,:]
ids = ids[1000:1999,:]

int_to_word = np.load('int_to_word10.npy')
word_to_int = np.load('word_to_int10.npy')

int_to_word = int_to_word.item()
word_to_int = word_to_int.item()

int_to_word[0] = '0'


# In[3]:

# In[]:
text = []
for quote in ids:
    for word in quote:
        if not word==0:
            text.append(word)

text = np.ndarray.flatten(np.asarray(text))


# In[4]:

# In[]: cut the text in semi-redundant sequences of maxlen characters
maxlen = 10
step = 1
seq = []
next_seq = []
quote_len = text.shape[0]

for i in range(0, quote_len - maxlen, step):
    seq.append(text[i: i + maxlen])
    next_seq.append(text[i + maxlen])

print('nb sequences:', len(seq))


# In[5]:

seq = np.asarray(seq)
next_seq = np.asarray(next_seq)

max_word = np.asarray(len(int_to_word))


# In[6]:

seq, next_seq, seq.shape, next_seq.shape, max_word


# In[7]:

int_to_word[13636]


# In[8]:

def relu_advanced(x):
    return K.relu(x, max_value=max_word)

from keras.utils import to_categorical


# In[9]:

# reshape X to be [samples, time steps, features]
X = np.reshape(seq, (len(seq), maxlen, 1))
# normalize
X = X / max_word
# one hot encode the output variable
#y = to_categorical(next_seq, num_classes= max_word)
y = np.reshape(next_seq,(next_seq.shape[0],1)) #/ max_word


# In[10]:

y.shape


# In[11]:

# build the model: LSTM
print('Build model...')
model = Sequential()

model.add(LSTM(256, input_shape=(
                            X.shape[1], #None,
                            X.shape[2]),
               return_sequences=True,
               #activation='sigmoid'
               #unroll=True
               ))


# model.add(Dropout(0.2))
model.add(LSTM(256))
# model.add(LSTM(y.shape[1]
#                ,activation='sigmoid'
#                ))

# model.add(Dropout(0.2))

model.add(Dense(32))

model.add(Dense(y.shape[1]
                ,activation=relu_advanced #'sigmoid'#relu_advanced # 0 to max_word
               ))

#model.add(Activation('softmax'))

#optimizer = RMSprop(lr=0.01)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(
    loss='mse',
#    loss= 'mean_squared_logarithmic_error',  #'sparse_categorical_crossentropy',
    optimizer=optimizer)


# In[12]:

X.shape


# In[13]:

# In[8]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # preds = preds[1:]
    # preds = np.asarray(preds).astype('float64')
    # preds = np.log(preds) / temperature
    # exp_preds = np.exp(preds)
    # preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)
    #return np.argmax(preds)
    return preds


# In[14]:

def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        print(sentence)
        generated.join([str([int_to_word[value]]).join(' ') for value in sentence])
        print('----- Generating with seed: %s'%[int_to_word[word] for word in sentence])
        sys.stdout.write(generated)

        for i in range(15):
            x_pred = np.reshape(sentence,(1, 
                                          -1, #maxlen, 
                                          1))
            x_pred = x_pred / max_word

            preds = model.predict(x_pred, verbose=0)
            preds = preds[0]
            #print(preds.shape)
            next_index = round(sample(preds, diversity)[0]) # 
            #print(next_index)
            next_char = int_to_word[next_index]

            generated.join(str(next_char))
            sentence = np.append(
                                 sentence[1:],   #sentence, 
                                 next_index)

            sys.stdout.write(next_char)
            sys.stdout.write(" ")
            sys.stdout.flush()
        print()


# In[15]:

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

filepath="trained_weights/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


# In[16]:

model.fit(X, y,
          batch_size=128,
          epochs=1000,
          #validation_split=0.05,
          callbacks=[print_callback, checkpoint])


# In[ ]:



