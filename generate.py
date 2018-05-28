import argparse
parser = argparse.ArgumentParser()
parser.add_argument("topics", help="select atleast 2 topics from 'death' ,'family', 'funny', 'freedom' , 'life' , 'love', 'happiness',  'science', 'success', 'politics'", type=str, nargs='+')
parser.add_argument("-s","--seed", help="input a custom seed sentence")
args = parser.parse_args()
if len(args.topics) < 2 and not args.seed:
        raise Exception("select atleast 2 topics from 'death' ,'family', 'funny', 'freedom' ,'life' , 'love', 'happiness',  'science', 'success', 'politics'")


import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from Model import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In[3]:

with open('data/all.txt','r') as quotefile:
    quotes = quotefile.readlines()


# In[4]:

t = Tokenizer(filters='')
t.fit_on_texts(quotes)
vocab_size = len(t.word_index) + 1


# In[12]:

index_word = np.load('data/index_word.npy')
index_word = index_word.item()


topics = args.topics

# In[13]:

## accomodate custom seed
model_topics = []
funny_doc = []
seedlen = 50
maxlen = 50
sentence=""

if not args.seed:
    with open('data/%s.txt'%topics[0],'r') as funnyfile:
        funnyquotes = funnyfile.readlines()

    encoded_docs = t.texts_to_sequences(funnyquotes)
    funny_doc = encoded_docs[0]
    start_index = np.random.randint(0, len(funny_doc) - seedlen - 1)
    sentence = funny_doc[start_index: start_index + seedlen]

    model_topics = topics[1:]
else:
    sentence = args.seed
    sentence = t.texts_to_sequences([word for word in sentence.split(' ')])
    sentence = list(filter(None, sentence))
    sentence = np.asarray(sentence).flatten()
    model_topics = topics


model_list = []
# ## Do for all docs except first
for topic in model_topics:
    model_funny = Model(vocab_size,topic)
    model = model_funny.load_model()
    model_list.append(model)



def on_epoch_end(sentence, model, maxlen = 10):
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

        # sys.stdout.write(next_char)
        if i % (maxlen // 4) == 0:
            sys.stdout.write("-")
        sys.stdout.flush()

    sys.stdout.write("\n")
    print('----- Input seed: %s'%original_sentence.split('.')[-1])
    print('----- Output: %s'%predicted.split('.')[0])
    sys.stdout.write("-----\n")
    return original_sentence.split('.')[-1] + predicted.split('.')[0]

for model in model_list:
    sentence = on_epoch_end(sentence,model,maxlen)
    sentence = np.asarray(t.texts_to_sequences([word for word in sentence.split(' ')][1:-1])).flatten()
    #sentence = sentence[maxlen:] #
