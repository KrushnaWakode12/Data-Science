#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


bike = pd.read_csv('C:\\Users\HP\Desktop\Ebooks\Work\IT Projects\Data Science\Python\Linear Regression - Bikeshare\Bikeshare.csv')


# In[3]:


bike.head(6)


# In[4]:


plt.scatter(bike['temp'], bike['count'], c=bike['temp'], cmap='Blues', alpha=0.3)
cbar=plt.colorbar(label='Temp')
plt.xlabel('temp') 
plt.ylabel('count');


# In[5]:


x=bike['datetime'].copy()
for i in range(10886):
    x[i]=x[i][0:10]

plt.scatter(x, bike['count'], c=bike['temp'],cmap='summer', alpha=0.3)
cbar=plt.colorbar(label='Temp')
plt.xlabel('datetime') 
plt.ylabel('count');


# In[7]:


bike.corr()


# In[8]:


x=bike['datetime'].copy()
for i in range(10886):
    x[i]=x[i][11:13]
bike['hour'] = x


# In[9]:


plt.scatter(bike[bike['workingday']==1]['hour'],bike[bike['workingday']==1]['count'], c=bike[bike['workingday']==1]['temp'], cmap= 'rainbow',alpha=0.5)
cbar=plt.colorbar(label='Temp on WD')
plt.xlabel('hour') 
plt.ylabel('count');


# In[10]:


plt.scatter(bike[bike['workingday']==0]['hour'],bike[bike['workingday']==0]['count'], c=bike[bike['workingday']==0]['temp'], cmap= 'rainbow',alpha=0.5)
cbar=plt.colorbar(label='Temp on non-WD')
plt.xlabel('hour') 
plt.ylabel('count');


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[12]:


temp_model = LinearRegression(fit_intercept=False)


# In[37]:


bike['hour'] = bike['hour'].astype(dtype='int')
a=('season','holiday','workingday','weather','temp','humidity','windspeed','hour')
X = bike.loc[:,a]
#Xt=bike['temp']
y=bike['count']

xtrain,xtest,ytrain, ytest = train_test_split(X,y,random_state=1)
temp_model.fit(xtrain,ytrain)
temp_model.normalize
xtest.index=range(2722)
ypred= temp_model.predict(xtest)


# In[43]:


temp_model.coef_


# In[45]:


temp_model.intercept_


# In[46]:


np.random.seed(1)
err = np.std([temp_model.fit(xtrain,ytrain).coef_ for i in range(1000)], 0)


# In[47]:


params = pd.Series(temp_model.coef_, index= X.columns)
print(pd.DataFrame({'effect':params.round(0) , 'error': err.round(0)}))


# In[ ]:




