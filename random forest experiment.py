
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
from os import listdir
a=pd.read_csv("/home/jash/Desktop/capstone/data_after_inception.csv")
a=pd.DataFrame(data=a,index=None)


# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split


# In[5]:



df_x=a.iloc[:,1:-1]
df_y=a.iloc[:,2049]
print df_y.head()


# In[9]:



x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=5)
Accuracy=[]
k=50
while (k<500):
    
    rf=RandomForestClassifier(n_estimators=k)
    rf.fit(x_train,y_train)
    pred=rf.predict(x_test)
    p=len(pred)
    s=y_test.values
    count=0
    for i in range(p):
        if pred[i]==s[i]:
            count=count+1.0
    m=(float(count/p))
    Accuracy.append(m)
                 
    print "Accuracy",k,"  is:", m
    k=k+50

