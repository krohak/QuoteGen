import os
import sys
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# In[3]:

with open('data/all.txt','r') as quotefile:
    quotes = quotefile.readlines()

if not os.path.exists('trained_weights'):
    os.mkdir('trained_weights')

# In[4]:

t = Tokenizer(filters='')
t.fit_on_texts(quotes)
vocab_size = len(t.word_index) + 1
print(vocab_size)


# In[12]:

embedding_matrix = np.load('data/embedding_matrix.npy')
embedding_matrix.shape

index_word = np.load('data/index_word.npy')
index_word = index_word.item()


topics = ['death' , 'family', 'freedom' , 'funny', 'life' , 'love', 'happiness', 'success', 'science', 'politics']
# ## Do for all docs
for topic in topics:

    # In[13]:

    with open('data/%s.txt'%topic,'r') as funnyfile:
        funnyquotes = funnyfile.readlines()


    # In[14]:

    encoded_docs = t.texts_to_sequences(funnyquotes)
    funny_doc = encoded_docs[0]


    # In[15]:

    maxlen = 100
    step = 1
    seq_funny = []
    next_seq_funny = []

    quote_len_funny = len(funny_doc)


    # In[16]:

    for i in range(0, quote_len_funny - maxlen, step):
        seq_funny.append(funny_doc[i: i + maxlen])
        next_seq_funny.append(funny_doc[i + maxlen])

    print('nb sequences:', len(seq_funny))

    seq_funny = np.asarray(seq_funny)
    next_seq_funny = np.asarray(next_seq_funny)


    # # Text Generation using Word Embeddings


    # In[20]:
    y = to_categorical(next_seq_funny, num_classes=vocab_size)
    X = seq_funny

    # In[26]:
    model = Sequential()
    e = Embedding( vocab_size, 100, weights=[embedding_matrix], trainable=True)
    model.add(e)

    # In[29]:

    model.add(LSTM(100))
    model.add(Dense(y.shape[1]))

    # In[30]:

    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    print(model.summary())


    # In[32]:

    def on_epoch_end(epoch, logs):

        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = np.random.randint(0, len(funny_doc) - maxlen - 1)
        sentence = funny_doc[start_index: start_index + maxlen]

        predicted = ''
        original_sentence = ''.join([str(index_word[word])+' ' for word in sentence])
        for i in range(maxlen):
            x_pred = np.reshape(sentence,(1, -1))

            preds = model.predict(x_pred, verbose=0)
            preds = preds[0]
            next_index =  np.argmax(preds)
            next_char = index_word[next_index]

            sentence = np.append(sentence, next_index)
            predicted = predicted + next_char + ' '

            if i % (maxlen // 4) == 0:
                sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("\n")
        print('----- Input seed: %s'%original_sentence.split('.')[-1])
        print('----- Output: %s'%predicted.split('.')[0])
        sys.stdout.write("-----\n")

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    # In[33]:
    filepath="trained_weights/QG-%s-{epoch:02d}-{loss:.4f}-{acc:.4f}.hdf5"%topic
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # In[ ]:

    model.fit(X, y, epochs=30, batch_size=24, callbacks=[checkpoint, print_callback])
