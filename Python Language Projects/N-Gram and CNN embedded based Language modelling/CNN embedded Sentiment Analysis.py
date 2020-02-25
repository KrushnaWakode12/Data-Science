#!/usr/bin/env python
# coding: utf-8

# In[41]:


from nltk.corpus import stopwords
import string
import re
from os import listdir
from collections import Counter
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model


# In[2]:


# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# In[25]:


# turn a doc into clean tokens
def clean_doc(doc, vocab):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [w for w in tokens if w in vocab]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# In[4]:


# load the document
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)


# In[6]:


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


# In[7]:


# load all docs in a directory
def process_docs(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)


# In[8]:


# define vocab
vocab = Counter()
process_docs('txt_sentoken/pos', vocab)
process_docs('txt_sentoken/neg', vocab)
print(len(vocab))
print(vocab.most_common(20))


# In[9]:


# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))


# In[10]:


# save list to file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

save_list(tokens, 'vocab.txt')


# In[11]:


vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())


# In[23]:


# load all docs in a directory
def process_dir(directory, vocab, is_train):
    documents = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc, vocab)
        documents.append(tokens)
    return documents


# In[24]:


# load and clean a dataset
def load_clean_dataset(vocab, is_train):
    neg = process_dir('txt_sentoken/neg', vocab, is_train)
    pos = process_dir('txt_sentoken/pos', vocab, is_train)
    docs = neg + pos
    labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
    return docs, labels


# In[14]:


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# In[17]:


# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
    encoded = tokenizer.texts_to_sequences(docs)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


# In[18]:


# define the model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # summarize defined model
    model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


# In[19]:


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())


# In[26]:


# load training data
train_docs, ytrain = load_clean_dataset(vocab, True)


# In[27]:


# create the tokenizer
tokenizer = create_tokenizer(train_docs)


# In[28]:


# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)


# In[32]:


# calculate the maximum sequence length
max_length = max([len(s) for s in train_docs])
print('Maximum length: %d' % max_length)


# In[33]:


# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)


# In[34]:


# define model
model = define_model(vocab_size, max_length)


# In[35]:


# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=2)


# In[36]:


test_docs, ytest = load_clean_dataset(vocab, False)


# In[39]:


# save the model
model.save('model.h5')


# In[38]:


Xtest = encode_docs(tokenizer, max_length, test_docs)


# In[42]:


# load the model
model = load_model('model.h5')
_, acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Train Accuracy: %f' % (acc*100))

_, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))


# In[46]:


# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
    line = clean_doc(review, vocab)
    padded = encode_docs(tokenizer, max_length, [line])
    yhat = model.predict(padded, verbose=2)
    percent_pos = yhat[0,0]
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


# In[47]:


# test positive text
text = 'Everyone will enjoy this film. I love it, recommended!'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))


# In[48]:


# test negative text
text = 'This is a bad movie. Do not watch it. It sucks.'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))


# In[ ]:




