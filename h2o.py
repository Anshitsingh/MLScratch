
# coding: utf-8

# In[1]:


import pandas as pd
a=pd.read_table("/home/jash/Desktop/petplan/Profitability1.txt", sep='|', lineterminator='\n', error_bad_lines=False)


# In[2]:


data=pd.DataFrame(data=a,index=None)
data = data.sample(frac=1).reset_index(drop=True)
import math
import numpy as np
data['LossRatio'].fillna(0, inplace=True)
data['L']=0
data.loc[data['LossRatio'] >.57, 'L'] = 1
data.to_csv("/home/jash/Desktop/petplan/Profitability.csv")
Y=data['L']
y=pd.DataFrame(Y)
x=pd.DataFrame(data)


# In[3]:


x.drop(['LossRatio'],axis=1,inplace=True)
x.drop(['CustomerNumber','PhoneNumber','Surname','GivenName','CustomerMailingAddr_Addr1','CustomerMailingAddr_City','CustomerMailingAddr_StateProvCd',
       'CustomerMailingAddr_PostalCode','PetId','StartDate','EndDate','LastPolicyRef'],axis=1,inplace=True)

p=pd.DataFrame(x)
p.drop(p.index[334476],inplace=True)
y.drop(y.index[334476],inplace=True)
p['TotalClaimsAmtPaid'].fillna(0, inplace=True)


# In[4]:


p.isnull().sum()


# In[5]:


p.drop(['ClaimNumber','ClaimAmount','Severity','ClaimDetails','ConditionGrp','Claimcodecategory','claimdurationInception'],axis=1,inplace=True)


# In[6]:


p.isnull().sum()


# In[7]:


p.drop(['Quadrant'],axis=1,inplace=True)


# In[8]:


p.dropna(how='any',inplace=True)


# In[9]:


import h2o
from h2o.estimators import H2ORandomForestEstimator
h2o.init()
df = h2o.H2OFrame(p)
y = 'L'
x=df.col_names
print df
x.remove(y)
df[y] = df[y].asfactor()
train, valid, test = df.split_frame(ratios=[.8,.1])
from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm_cv3 = H2OGradientBoostingEstimator(nfolds=3)
gbm_cv3.train(x=x, y=y, training_frame=train)
## Getting all cross validated models 
all_models = gbm_cv3.cross_validation_models()
print("Total cross validation models: " + str(len(all_models)))


# In[10]:


gbm_cv3.confusion_matrix()
performance = gbm_cv3.model_performance(test_data=test)
print performance


# In[12]:


import matplotlib.pyplot as plt
gbm_cv3.varimp_plot()

