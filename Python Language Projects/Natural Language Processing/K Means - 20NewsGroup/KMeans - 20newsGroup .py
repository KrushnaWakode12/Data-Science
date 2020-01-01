#!/usr/bin/env python
# coding: utf-8

# **---Import Necessary Libraries**

# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# **---Import Data**

# In[3]:


from sklearn.datasets import fetch_20newsgroups


# In[4]:


all_df = fetch_20newsgroups(subset='all')


# **---Check Details of data such as length and Target names**

# In[5]:


print(len(all_df.filenames))


# In[6]:


print(all_df.target_names)


# **---Subset only perticular domain related dataset**

# In[7]:


groups = ['comp.graphics', 'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
'comp.windows.x', 'sci.space']

train = fetch_20newsgroups(subset='train', categories=groups)


# In[8]:


print(len(train.filenames))


# In[9]:


test = fetch_20newsgroups(subset='test', categories=groups)


# In[10]:


print(len(test.filenames))


# **---Import TfidfVectorizer and transform training data**

# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:


vect = TfidfVectorizer(min_df=10, max_df=0.5, decode_error='ignore', stop_words='english')


# In[13]:


vec_df = vect.fit_transform(train.data)


# In[14]:


num_samp, num_feature = vec_df.shape


# In[15]:


print('#samples: ', num_samp,'  #features: ',num_feature)


# **---Import K Means clustering model and train it with vectorized data**

# In[16]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=20, init='random', n_init=1, verbose=1, random_state=3)


# In[17]:


km.fit(vec_df)


# **---Print clusters and words belong to that cluster**

# In[35]:


order_cen = km.cluster_centers_.argsort()[:,::-1]
terms = vect.get_feature_names()

for i in range(20):
    print('cluster %d: ' %i)
    for x in order_cen[i,:20]:
        print('%s' %terms[x])
    print('\n')


# **---Evaluate classification model performance**

# In[20]:


from sklearn import metrics

print('Homogenity: %0.3f' %metrics.homogeneity_score(train.target, km.labels_))
print('Completeness: 0.3%f' %metrics.completeness_score(train.target, km.labels_))
print('V-measure: 0.3%f' %metrics.v_measure_score(train.target,km.labels_))


# **---Take Input from User and show relevant post related to it**

# In[30]:


post = str(input('Enter Your Query:\n'))
newv = vect.transform(post.split())


# In[57]:


post_label = km.predict(newv)
a=np.argmax((pd.value_counts(post_label)))


# In[32]:


data = pd.DataFrame()
data['data_index'] = pd.DataFrame(train.data).index.values
data['cluster'] = km.labels_
data['Str'] = pd.DataFrame(train.data)


# In[58]:


possible = data[data['cluster'] == a].head()


# In[59]:


print('Related Query:\n')

print(possible['Str'][0])


# In[ ]:





# In[ ]:




