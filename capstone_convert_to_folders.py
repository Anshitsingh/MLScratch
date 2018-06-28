
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import tensorflow as tf


# In[2]:


a=pd.read_csv("/home/jash/Desktop/capstone/labels_new.csv")


# In[3]:


a.head()


# In[4]:


labels=[]

for j in range(len(a)):
    b=a.iloc[j]['breed']
    labels.append(b)
    

unique_labels = list(set(labels))


# In[5]:


len(unique_labels)


# In[6]:


'/home/jash/Desktop/capstone/aa/'


# In[7]:


for index,row in a.iterrows():
    if not tf.gfile.Exists('/home/jash/Desktop/capstone/sorted_class/%s'%(row['breed'])):
        tf.gfile.MkDir('/home/jash/Desktop/capstone/sorted_class/%s'%(row['breed']))
#     print('train/%s.jpg'%(row['id']),'dog_train/%s'%(row['breed']))
    tf.gfile.Copy('/home/jash/Desktop/capstone/train/%s.jpg'%(row['id']),'/home/jash/Desktop/capstone/sorted_class/%s/%s.jpg'%(row['breed'],row['id']),True)

