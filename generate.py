import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from Model import Model

# In[3]:

with open('data/all-punctuation.txt','r') as quotefile:
    quotes = quotefile.readlines()

    
# In[4]:

t = Tokenizer()
t.fit_on_texts(quotes)
vocab_size = len(t.word_index) + 1


# In[12]:

index_word = np.load('index_word_punc.npy')
index_word = index_word.item()


topics = [ #'death' , 'family', 'freedom' , 'funny', 'life' ,
    	#'love', 
	#'happiness', 'death',
	'success',  'love', 'death'
	#'science', 'politics'
]

# In[13]:
with open('data/%s-punctuation.txt'%topics[0],'r') as funnyfile:
    funnyquotes = funnyfile.readlines()

encoded_docs = t.texts_to_sequences(funnyquotes)
funny_doc = encoded_docs[0]

model_list = []


# ## Do for all docs
for topic in topics[1:]:
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


def on_epoch_end(sentence, model, maxlen = 10):
    print()
    print('----- Generating text:')


    for diversity in [1.0]: #0.2, 0.5, 1.2
        print('----- diversity:', diversity)

        generated = ''
        print(sentence)
        generated.join([str([index_word[value]]).join(' ') for value in sentence])
        print('----- Generating with seed: %s'%[index_word[word] for word in sentence])

        for i in range(maxlen):
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

        
seedlen = 10
maxlen = 10
start_index = np.random.randint(0, len(funny_doc) - seedlen - 1)
sentence = funny_doc[start_index: start_index + seedlen]
#sentence = [2, 13, 148, 5, 2] # to be happy is to

for model in model_list:
    sentence = on_epoch_end(sentence,model,maxlen)
    #sentence = sentence[maxlen:] #
