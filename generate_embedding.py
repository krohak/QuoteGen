import sys
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding
from keras.optimizers import Adam

# In[3]:

with open('data/all.txt','r') as quotefile:
    quotes = quotefile.readlines()

    
# In[4]:

t = Tokenizer()
t.fit_on_texts(quotes)
vocab_size = len(t.word_index) + 1



# In[12]:

index_word = np.load('index_word.npy')
index_word = index_word.item()


topics = [ #'death' , 'family', 'freedom' , 'funny', 'life' ,
    	#'love', 
	'happiness', 
	#'success', 'science', 'politics'
]


# ## Do for all docs
for topic in topics:

    # In[13]:

    with open('data/%s.txt'%topic,'r') as funnyfile:
        funnyquotes = funnyfile.readlines()


    # In[14]:

    encoded_docs = t.texts_to_sequences(funnyquotes)
    funny_doc = encoded_docs[0]

    
    # In[15]:
    maxlen = 10
    
    # # Text Generation using Word Embeddings


    # In[20]:

    e = Embedding( vocab_size, 100, 
        #weights=[embedding_matrix], input_length=maxlen, trainable=False
        )


    # In[27]:

    model = Sequential()
    model.add(e)

    model.add(LSTM(100))
    model.add(Dense(vocab_size))
    

    # In[30]:

    model.add(Activation('softmax'))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    
    filename = "QG-weights-%s.hdf5"%topic
    model.load_weights(filename)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    print(model.summary())


    # In[32]:

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

            for i in range(20):
                x_pred = np.reshape(sentence,(1, -1))

                preds = model.predict(x_pred, verbose=0)
                preds = preds[0]
                next_index = sample(preds, diversity)
                next_char = index_word[next_index]

                generated.join(str(next_char))
                sentence = np.append(sentence, next_index)

                sys.stdout.write(next_char)
                sys.stdout.write(" ")
                sys.stdout.flush()
            print()

    on_epoch_end(1,1)
