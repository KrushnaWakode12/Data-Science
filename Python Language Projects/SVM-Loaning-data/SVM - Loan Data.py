#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv('loan_data.csv')


# In[6]:


df.describe()


# In[17]:


df.isna()


# In[18]:


df.dtypes


# In[32]:


plt.scatter(df['int.rate'],df['fico'], c=df['not.fully.paid'],alpha=0.25)


# In[57]:


df['purpose'], a =df['purpose'].factorize()[0], df['purpose'].factorize()[1]

X= df.iloc[:,:13]
Y=df.iloc[:,13]


# In[58]:


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X,Y, random_state=1)


# In[59]:


from sklearn.svm import SVC
model=SVC(kernel='rbf', class_weight='balanced')


# In[60]:


model.fit(xtrain,ytrain)


# In[61]:


ypred=model.predict(xtest)


# In[62]:


from sklearn.metrics import classification_report
print(classification_report(ytest,ypred))


# In[73]:


from sklearn.metrics import confusion_matrix
mat=confusion_matrix(ytest,ypred)
ytest.index=range(2395)


# In[86]:


a=pd.DataFrame((ytest,ypred))


# In[89]:


a=a.T


# In[91]:


a.columns = ['Actual','predicted']


# In[93]:


a.to_csv('outcome.csv')


# In[ ]:




