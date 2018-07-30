
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("/home/jash/Desktop/petplan/Profitability1.txt", sep='\t', lineterminator='\n')


# In[2]:


df.iloc[267910][0]


# In[3]:


df.iloc[267909][0]


# In[4]:


df.iloc[299539][0]


# In[5]:


a=pd.read_table("/home/jash/Desktop/petplan/Profitability1.txt", sep='|', lineterminator='\n', error_bad_lines=False)


# In[6]:


a.head()


# In[7]:


data=pd.DataFrame(data=a,index=None)


# In[8]:


data.shape


# In[9]:


data['L']=0


# In[10]:


data.loc[data['LossRatio'] >.57, 'L'] = 1


# In[11]:


data.head()


# In[12]:


data.to_csv("/home/jash/Desktop/petplan/Profitability.csv")


# In[13]:


y=data['L']


# In[14]:


data.drop(['LossRatio'],axis=1,inplace=True)


# In[15]:


data.drop(['L'],axis=1,inplace=True)


# In[16]:


x=data


# In[17]:


data['HighTenure-HighLR']=0
data['HighTenure-LowLR']=0
data['LowTenure-HighLR']=0
data['LowTenure-LowLR']=0


# In[18]:


data.loc[data['Quadrant']=='HighTenure-HighLR', 'HighTenure-HighLR'] = 1
data.loc[data['Quadrant']=='HighTenure-LowLR', 'HighTenure-LowLR'] = 1
data.loc[data['Quadrant']=='LowTenure-HighLR', 'LowTenure-HighLR'] = 1
data.loc[data['Quadrant']=='LowTenure-LowLR', 'LowTenure-LowLR'] = 1


# In[19]:


data.head()


# In[20]:


data.drop(['Quadrant'],axis=1,inplace=True)


# In[21]:


x.head()
x.to_csv("/home/jash/Desktop/petplan/xonly.csv")


# In[22]:


x.drop(['CustomerNumber','PhoneNumber','Surname','GivenName','CustomerMailingAddr_Addr1','CustomerMailingAddr_City','CustomerMailingAddr_StateProvCd',
       'CustomerMailingAddr_PostalCode','PetId','Country'],axis=1,inplace=True)


# In[23]:


x.columns


# In[24]:


x.head()


# In[25]:


x.to_csv("/home/jash/Desktop/petplan/xonly_edited.csv")


# In[26]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[28]:


X_train.shape


# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[30]:



clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[31]:


import numpy as np
from sklearn.decomposition import PCA


# In[32]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[33]:


X_train.head()


# In[34]:


X_train.columns


# In[35]:


import h2o
from h2o.estimators import H2ORandomForestEstimator


# In[ ]:


h2o.init()

