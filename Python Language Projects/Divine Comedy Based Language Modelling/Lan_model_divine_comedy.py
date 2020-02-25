#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import re
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from random import randint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# In[2]:


# load doc into memory
def load_doc(filename):
    file = open(filename, 'r', encoding='utf-8')
    text = file.read()
    file.close()
    return text


# In[3]:


# turn a doc into clean tokens
def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


# In[4]:


# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# In[5]:


# load document
filename = 'Divine Comedy rough.txt'
doc = load_doc(filename)
print(doc[:200])


# In[6]:


# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# In[7]:


# organize into sequences of tokens
length = 50 + 1
sequences = list()


# In[8]:


for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)


# In[9]:


print('Total Sequences: %d' % len(sequences))


# In[10]:


# save sequences to file
out_filename = 'divine_comedy_sequences.txt'
save_doc(sequences, out_filename)


# In[11]:


# load
in_filename = 'divine_comedy_sequences.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')


# In[12]:


# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


# In[13]:


# vocabulary size
vocab_size = len(tokenizer.word_index) + 1


# In[14]:


# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


# In[24]:


# define the model
def define_model(vocab_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


# In[25]:


model = define_model(vocab_size, seq_length)


# In[15]:


model = load_model('model.h5');


# In[ ]:


model.fit(X, y, batch_size=128, epochs=10)


# In[58]:


model.save('model.h5')


# In[62]:


seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')


# In[63]:


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break

        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


# In[64]:


# generate new text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)


# In[ ]:





# In[ ]:




