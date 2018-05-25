import numpy as np
import sys
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


    
# In[4]:

t = Tokenizer()
t.fit_on_texts(quotes)
vocab_size = len(t.word_index) + 1



# In[12]:

embedding_matrix = np.load('embedding_matrix.npy')
embedding_matrix.shape


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


    # In[22]:

    X = seq_funny


    # In[25]:

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)



    # In[26]:

    e = Embedding(
        vocab_size, 
        100, 
        weights=[embedding_matrix], 
        #input_length=maxlen, 
        #trainable=False
        )


    # In[27]:

    model = Sequential()
    model.add(e)


    # In[28]:

    #model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    #model.add(MaxPooling1D(pool_size=3))


    # In[29]:

    #model.add(LSTM(100, return_sequences=True)) #
    model.add(LSTM(100))

    #model.add(Dense(100, activation='relu'))
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

    index_word = np.load('index_word.npy')
    index_word = index_word.item()

    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        # preds = preds[1:]
        # preds = np.asarray(preds).astype('float64')
        # preds = np.log(preds) / temperature
        # exp_preds = np.exp(preds)
        # preds = exp_preds / np.sum(exp_preds)
        # probas = np.random.multinomial(1, preds, 1)
        return np.argmax(preds)


    def on_epoch_end(epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = np.random.randint(0, len(funny_doc) - maxlen - 1)
        for diversity in [0.2]: # 0.5, 1.0, 1.2
            print('----- diversity:', diversity)

            generated = ''
            sentence = funny_doc[start_index: start_index + maxlen]
            print(sentence)
            generated.join([str([index_word[value]]).join(' ') for value in sentence])
            print('----- Generating with seed: %s'%[index_word[word] for word in sentence])
            #sys.stdout.write(generated)

            for i in range(20):
                x_pred = np.reshape(sentence,(1, -1 #maxlen
                            ))

                preds = model.predict(x_pred, verbose=0)
                preds = preds[0]
                # print(preds.shape)
                next_index = sample(preds, diversity)
                #print(next_index)
                next_char = index_word[next_index]

                generated.join(str(next_char))
                sentence = np.append(sentence #[1:]
                        ,next_index)

                sys.stdout.write(next_char)
                sys.stdout.write(" ")
                sys.stdout.flush()
            print()



    # In[33]:
    filepath="QG-weights-%s-{epoch:02d}-{loss:.4f}-{accuracy:.4f}.hdf5"%topic
    checkpoint = ModelCheckpoint(filepath, monitor='loss, accuracy', verbose=1, save_best_only=True, mode='min')

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


    # In[ ]:

    model.fit(X_train, y_train, validation_data=(X_test, y_test), 
              epochs=50, 
              batch_size=1024,
              callbacks=[checkpoint, print_callback])
