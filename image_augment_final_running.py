
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os


# In[2]:


a=pd.read_csv("/home/jash/Desktop/capstone/labels.csv")


# In[3]:


print len(a)


# In[4]:


x=a.groupby("breed")


# In[5]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.image as mpimg



# In[6]:


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# In[7]:


def transform_image(img,ang_range,shear_range,trans_range,brightness=0):

    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape    
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


# In[8]:


for j in range(len(a)):
    
    x=a.iloc[j]['id']+'.jpg'
    data_path='/home/jash/Desktop/capstone/train_new/'
    image_path = os.path.join(data_path,x)
    t=len(a)
    inputa = cv2.imread(image_path)
    
    if inputa is not None:
        
        p='/home/jash/Desktop/capstone/train/'+x
        image = inputa

        
        for i in range(3):
            print j,"j",i,"i"
            img = transform_image(image,20,10,5,brightness=1) 
            print img.shape,i
            k=i+1
            f=a.iloc[j]['id']
            g=a.iloc[j]['breed']
            cv2.imwrite('/home/jash/Desktop/capstone/train/'+f+str(k)+str(j)+'.jpg',img)
            a=a.append({'id':f+str(k)+str(j),'breed':a.iloc[j]['breed']},ignore_index=True)
            
    else:
        print 'image didnt load'

    

    






# In[9]:


a.to_csv("/home/jash/Desktop/capstone/labels_new.csv", columns=("id","breed"))


# In[10]:


print len(a)

