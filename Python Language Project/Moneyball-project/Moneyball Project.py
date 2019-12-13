#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import pandas as pd


# In[5]:


batting = pd.read_csv('C:\\Users\HP\Desktop\Ebooks\Work\IT Projects\Data Science\Python\Moneyball Project\Batting.csv')
sal = pd.read_csv('C:\\Users\HP\Desktop\Ebooks\Work\IT Projects\Data Science\Python\Moneyball Project\Salaries.csv')


# In[6]:


batting['BA'] = batting['H'] / batting['AB']


# In[7]:


batting['OBP']=(batting['H'] + batting['BB'] + batting['HBP'])/(batting['AB'] + batting['BB'] + batting['HBP'] + batting['SF'])


# In[8]:


batting['1B']=batting['H'] - batting['2B'] - batting['3B'] - batting['HR']


# In[9]:


batting['SLG'] = ((batting['1B']) + (2 * batting['2B']) + (3 * batting['3B']) + (4 * batting['HR']) ) / batting['AB']


# In[10]:


batting.describe()


# In[11]:


sal.describe()


# In[12]:


batting = batting[batting['yearID'] >= 1985]


# In[13]:


batting.describe()


# In[16]:


combo = pd.merge(batting,sal, on = ['playerID', 'yearID'])


# In[17]:


combo.describe()


# In[50]:


a = combo[combo['playerID'] == 'giambja01']


# In[51]:


b = combo[combo['playerID'] == 'damonjo01']


# In[57]:


c= combo[combo['playerID'] == 'saenzol01']


# In[61]:


a = a.append(b)
a = a.append(c)


# In[62]:


a


# In[65]:


lost_players = a[a['yearID'] == 2001]


# In[66]:


lost_players


# In[67]:


avail_players = combo[combo['yearID'] == 2001]


# In[68]:


avail_players


# In[69]:


plt.scatter(avail_players['OBP'],avail_players['salary'])


# In[74]:


avail_players = avail_players[avail_players['OBP'] > 0]


# In[75]:


avail_players = avail_players[avail_players['salary'] < 8000000]


# In[76]:


avail_players = avail_players[avail_players['AB'] >= 500]


# In[77]:


avail_players


# In[96]:


avail_players = avail_players.sort_values(by='OBP', ascending= False)


# In[97]:


avail_players.head(n=10)


# In[104]:


possible = avail_players[['playerID','OBP','AB','salary']].head(n=10)


# In[114]:


possible = possible.sort_values(by='salary')


# In[115]:


possible


# In[116]:


possible.iloc[[0,1,3],:]


# In[ ]:




