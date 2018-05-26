import sys
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

from keras.preprocessing.text import Tokenizer
from Model import Model
'''
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Embedding
from keras.optimizers import Adam
'''

# In[3]:

with open('data/all.txt','r') as quotefile:
    quotes = quotefile.readlines()

    
# In[4]:

t = Tokenizer()
t.fit_on_texts(quotes)
vocab_size = len(t.word_index) + 1

maxlen = 10

# In[12]:

index_word = np.load('index_word.npy')
index_word = index_word.item()


topics = [ #'death' , 'family', 'freedom' , 'funny', 'life' ,
    	#'love', 
	'happiness', 'death'
	#'success', 'science', 'politics'
]

# In[13]:
with open('data/%s.txt'%topics[0],'r') as funnyfile:
    funnyquotes = funnyfile.readlines()

encoded_docs = t.texts_to_sequences(funnyquotes)
funny_doc = encoded_docs[0]

model_list = []


# ## Do for all docs
for topic in topics:
    model_funny = Model(vocab_size,topic)
    model = model_funny.load_model()
    model_list.append(model)


    
# # Text Generation using Word Embeddings
def sample(preds, temperature=1.0):
# helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs, sentence, model):
    print()
    print('----- Generating text after Epoch: %d' % epoch)


    for diversity in [1.0]: #0.2, 0.5, 1.2
        print('----- diversity:', diversity)

        generated = ''
        print(sentence)
        generated.join([str([index_word[value]]).join(' ') for value in sentence])
        print('----- Generating with seed: %s'%[index_word[word] for word in sentence])

        for i in range(20):
            x_pred = np.reshape(sentence,(1, -1))

            preds = model.predict(x_pred, verbose=0)
            preds = preds[0]
            next_index = np.argmax(preds) #sample(preds,diversity)
            next_char = index_word[next_index]

            generated.join(str(next_char))
            sentence = np.append(sentence, next_index)

            sys.stdout.write(next_char)
            sys.stdout.write(" ")
            sys.stdout.flush()
    print()        
    return sentence

        
seed_len = 10        
start_index = np.random.randint(0, len(funny_doc) - seed_len - 1)
sentence = funny_doc[start_index: start_index + seed_len]

for model in model_list:
    print(model)
    sentence = on_epoch_end(1,1,sentence,model)
