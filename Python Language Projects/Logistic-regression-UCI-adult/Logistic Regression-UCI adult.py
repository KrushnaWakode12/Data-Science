#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df=pd.read_csv('Adult_sal.csv')


# In[14]:


df


# In[15]:


df.describe()


# In[16]:


df=df.iloc[:,1:]
df


# In[24]:


print(df['type_employer'].describe())
print(df['type_employer'].unique())


# In[37]:


for i in enumerate(df['type_employer']):
    if (i[1]=='Never-worked') | (i[1]=='Without-pay'):
        df['type_employer'][i[0]]='Unemployed'

print(df['type_employer'].describe())
print(df['type_employer'].unique())


# In[38]:


for i in enumerate(df['type_employer']):
    if (i[1]=='State-gov') | (i[1]=='Local-gov'):
        df['type_employer'][i[0]]='SL-gov'
    elif (i[1]=='Self-emp-not-inc') | (i[1]=='Self-emp-inc'):
        df['type_employer'][i[0]]='self-emp'

print(df['type_employer'].describe())
print(df['type_employer'].unique())


# In[39]:


print(df['marital'].describe())
print(df['marital'].unique())


# In[41]:


for i in enumerate(df['marital']):
    if i[1] in ['Separated','Divorced','Widowed']:
        df['marital'][i[0]]='Not-married'
    elif i[1] =='Never-married':
        df['marital'][i[0]]='Never-married'
    else:
        df['marital'][i[0]]='Married'

print(df['marital'].describe())
print(df['marital'].unique())


# In[42]:


print(df['country'].describe())
print(df['country'].unique())


# In[43]:


Asia =['China','Hong','India','Iran','Cambodia','Japan', 'Laos' ,
          'Philippines' ,'Vietnam' ,'Taiwan', 'Thailand']

North_America = ['Canada','United-States','Puerto-Rico']

Europe =['England' ,'France', 'Germany' ,'Greece','Holand-Netherlands','Hungary',
            'Ireland','Italy','Poland','Portugal','Scotland','Yugoslavia']

Latin_and_South_America = ['Columbia','Cuba','Dominican-Republic','Ecuador',
                             'El-Salvador','Guatemala','Haiti','Honduras',
                             'Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru',
                            'Jamaica','Trinadad&Tobago']
Other = ['South']


# In[44]:


for i in enumerate(df['country']):
    if i[1] in Asia:
        df['country'][i[0]]='Asia'
    elif i[1] in North_America:
        df['country'][i[0]]='North america'
    elif i[1] in Europe:
        df['country'][i[0]]='Europe'
    elif i[1] in Latin_and_South_America:
        df['country'][i[0]]='South america'
    else:
        df['country'][i[0]]='Other'

print(df['country'].describe())
print(df['country'].unique())


# In[59]:


from numpy import NaN
df['type_employer'][df['type_employer']=='?'] = NaN


# In[60]:


df=df.dropna()


# In[70]:


df['income'], inc=df['income'].factorize()


# In[73]:


df['marital'],mari=df['marital'].factorize()
df['occupation'], occ=df['occupation'].factorize()
df['type_employer'], employer= df['type_employer'].factorize()
df['country'], cntry = df['country'].factorize()

df['sex'], sx = df['sex'].factorize()


# In[78]:


df['race']=df['race'].factorize()[0]
df['relationship'] = df['relationship'].factorize()[0]
df['education'] = df['education'].factorize()[0]
df


# In[79]:


from sklearn.model_selection import train_test_split

x=df.iloc[:,:14]
y=df['income']
xtrain,xtest,ytrain,ytest= train_test_split(x,y, random_state=1)



from sklearn.linear_model import LogisticRegression


# In[80]:


model = LogisticRegression()
model.fit(xtrain,ytrain)


# In[81]:


ypred=model.predict(xtest)


# In[88]:


model.coef_


# In[91]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))


# In[93]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,ypred))


# In[118]:


a=pd.DataFrame((ytest,ypred))
a=a.T
a.coulmns = ['actual','predicted']
a.to_csv('outcome.csv')


# In[ ]:





# In[ ]:




