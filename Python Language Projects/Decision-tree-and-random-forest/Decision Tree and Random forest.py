#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('C:\\Users\HP\Desktop\Ebooks\Work\IT Projects\Data Science\Python\Decision Tree and Random Forest - College ISLR\college.csv')


# In[11]:


df


# In[68]:


import matplotlib.colors as cl
cMap=cl.ListedColormap(['Green','Orange'])
plt.scatter(df['Room.Board'],df['Grad.Rate'], c=df['Private'].factorize()[0], cmap=cMap)
a=plt.colorbar()
a.set_label('Private', rotation=90)
a.ax.set_yticklabels([' ','Yes',' ',' ','No',' ']);


# In[71]:


df[df['Grad.Rate'] > 100] = 100


# In[101]:


from sklearn.model_selection import train_test_split
dfs =df.iloc[:,1:]
x=dfs.loc[:,dfs.columns != 'Private']
y=dfs['Private'].factorize()[0]
xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=1)


# In[102]:


from sklearn.ensemble import RandomForestClassifier


# In[103]:


model = RandomForestClassifier(n_estimators=100)


# In[104]:


model.fit(xtrain,ytrain)


# In[105]:


ypred = model.predict(xtest)


# In[106]:


from sklearn import metrics
print(metrics.classification_report(ypred,ytest))


# In[117]:


mat=metrics.confusion_matrix(ytest,ypred)
import seaborn as sns
sns.heatmap(mat.T,square=True,annot=True,fmt='d', cbar=False)
plt.xlabel('True Value')
plt.ylabel('Predicted Value');
plt.savefig('outmat.jpeg')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




