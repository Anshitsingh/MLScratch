
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from os import listdir
a=pd.read_csv("/home/jash/Desktop/capstone/labels_new.csv")


# In[2]:



result_array = np.empty((0, 2049))
location = '/home/jash/Desktop/capstone/bottleneck'
folder_list = listdir(location)
for folder in folder_list:
   folder_loc = location+'/'+folder
   file_list = os.listdir(folder_loc)
   
   for i in file_list:
      
       
       file_list_loc = folder_loc+'/'+i
       data1 = pd.read_csv(file_list_loc, header= None )
       data1['labels']=str(folder)
       result_array = np.append(result_array,data1,axis= 0)
        
       
       
   print('Done '+str(folder))





# In[4]:



from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split


# In[11]:


a=pd.DataFrame(data=result_array,index=None)


# In[6]:


data=a
df_x=data.iloc[:,0:-1]
df_y=data.iloc[:,2048]



# In[7]:


data.to_csv("/home/jash/Desktop/capstone/data_after_inception.csv")


# In[8]:



x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
y_train.head()

rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)


pred=rf.predict(x_test)


# In[9]:


s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1.0
print("count:",count)
print("Total",len(pred))
print("Accuracy:",float(count/len(pred)))


# In[10]:


rf.score(x_train.values,y_train.values)

