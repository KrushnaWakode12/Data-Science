#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences


# In[1]:


# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# In[2]:


# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)


# In[3]:


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# In[4]:


# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)


# In[5]:


# organize into sequences of characters
length = 10
sequences = list()


# In[7]:


for i in range(length, len(raw_text)):
    # select sequence of tokens
    seq = raw_text[i-length:i+1]
    sequences.append(seq)
print('Total Sequences: %d' % len(sequences))


# In[8]:


# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)


# In[9]:


# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')


# In[10]:


chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))


# In[12]:


sequences = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)


# In[13]:


# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)


# In[16]:


sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]


# In[18]:


# separate into input and output
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)


# In[19]:


# define the model
def define_model(X):
    model = Sequential()
    model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(vocab_size, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize defined model
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


# In[20]:


# define model
model = define_model(X)


# In[21]:


# fit model
model.fit(X, y, epochs=100, verbose=2)


# In[22]:


# save the model to file
model.save('model_lan_rhyme.h5')


# In[23]:


# save the mapping
dump(mapping, open('mapping_lan_rhyme.pkl', 'wb'))


# In[40]:


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1,encoded.shape[1],38)
        yhat = model.predict_classes(encoded, verbose=0)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += out_char
    return in_text


# In[41]:


# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 20))


# In[42]:


# test mid-line
print(generate_seq(model, mapping, 10, 'king was i', 20))


# In[43]:


# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))


# In[ ]:




