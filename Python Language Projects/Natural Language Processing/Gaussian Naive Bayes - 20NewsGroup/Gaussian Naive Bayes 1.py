#!/usr/bin/env python
# coding: utf-8

# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[3]:


from sklearn.datasets import make_blobs


# In[4]:


X , y = make_blobs(centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:,0],X[:,1],c=y, s=50, cmap='RdBu')


# In[5]:


from sklearn.naive_bayes import GaussianNB


# In[6]:


model = GaussianNB()


# In[7]:


model.fit(X,y)


# In[8]:


rng = np.random.RandomState(0)
Xnew= [-6,-14] + [14,18]*rng.rand(2000,2)
ynew= model.predict(Xnew)


# In[9]:


plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c=ynew , alpha=0.2, cmap='RdBu', s=20)
plt.axis(lim);


# In[10]:


yprob = model.predict_proba(Xnew)


# In[11]:


yprob


# In[12]:


yprob.round(2)


# In[13]:


from sklearn.datasets import fetch_20newsgroups


# In[14]:


data=fetch_20newsgroups()


# In[15]:


data.target_names


# In[16]:


cat= ['talk.religion.misc','soc.religion.christian','sci.space','comp.graphics']
train = fetch_20newsgroups(subset='train', categories = cat)
test = fetch_20newsgroups(subset = 'test', categories = cat)


# In[17]:


print(train.data[5])


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# In[19]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[20]:


model.fit(train.data, train.target)
labels = model.predict(test.data)


# In[22]:


from sklearn.metrics import confusion_matrix
mat= confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square = True, annot = True, fmt ='d', cbar = False, xticklabels= train.target_names, yticklabels= train.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


# In[24]:


def predict_cat(s, train = train, model = model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# In[26]:


predict_cat('sending a payload to ISS')


# In[27]:


predict_cat('discussing islam vs atheism')


# In[28]:


predict_cat('determining screen resolution')


# In[ ]:




