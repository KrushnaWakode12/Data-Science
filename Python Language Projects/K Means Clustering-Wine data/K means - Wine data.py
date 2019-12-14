#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


dfr= pd.read_csv('winequality-red.csv', sep=';')


# In[8]:


dfw=pd.read_csv('winequality-white.csv',sep=';')


# In[14]:


dfr['label']='red'
dfw['label']='white'


# In[17]:


df=pd.concat([dfr,dfw])
df


# In[22]:


df.describe()


# In[39]:


import matplotlib.colors as cl
cMap=cl.ListedColormap('Gray','Red');
plt.scatter(df['citric acid'],df['residual sugar'], c=df['label'].factorize()[0], cmap=cMap, alpha=0.25)


# In[40]:


plt.scatter(df['volatile acidity'], df['residual sugar'], c=df['label'].factorize()[0], cmap=cMap, alpha=0.25)


# In[45]:


x=df.iloc[:,:12]
y=df['label']


# In[48]:


from sklearn.cluster import KMeans


# In[49]:


model=KMeans(n_clusters=2)


# In[50]:


model.fit(x)


# In[54]:


print(model.cluster_centers_)


# In[55]:


ypred=model.predict(x)


# In[62]:


plt.scatter(df['volatile acidity'], df['residual sugar'], c=ypred, s=50,cmap=cMap, alpha=0.25)
centers=model.cluster_centers_
plt.scatter(centers[:,1], centers[:,0], c='black', s=70)


# In[67]:


from sklearn.metrics import confusion_matrix

ytest=y.factorize()[0]
mat=confusion_matrix(ytest,ypred)


# In[70]:


from seaborn import heatmap
print(heatmap(mat.T,annot=True,fmt='d',cbar=False,square=True))


# In[81]:


a=pd.DataFrame((ytest,ypred))
a=a.T
a.columns=['actual','predicted']


# In[82]:


a.to_csv('outcome.csv')


# In[ ]:




