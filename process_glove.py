import numpy as np
from numpy import asarray

# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


np.save('embeddings_index.npy',embeddings_index)
