#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[58]:


img = cv2.imread('G:326051_16.jpg')


# In[59]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[60]:


faces = face_cas.detectMultiScale(gray)


# In[61]:


for (x,y,w,h) in faces:
    #cv2.rectangle(img,(x,y),(x+h,y+w),(255,255,255), 1)
    img_crop = img[x:x+h, y:y+w]


# In[ ]:


cv2.imshow('Image',img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


faces.view()


# In[8]:


cv2.imshow('ad',img)


# In[56]:





# In[ ]:




