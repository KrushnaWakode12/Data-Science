#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[1]:


cap = cv2.VideoCapture(0)


# In[4]:


while True:
    x, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.2, 6)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)
        
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k==27:
        cv2.destroyAllWindows()
        break

        
cap.release()


# In[ ]:




