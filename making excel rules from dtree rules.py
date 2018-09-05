
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
a=pd.read_csv("E:/Petplan/rules.csv")
b= pd.DataFrame(columns=['LossRatio'])
for i in range(len(a)):
    j=0
    while((j<len(a.columns))):
        if(str(a.ix[i][j])=='is'):
            b.loc[i,str(a.ix[i][j-1])]=str(a.ix[i][j+1])
        j=j+1
    b.loc[i,'LossRatio']=a.loc[i,'LossRatio']
                
b=b.fillna('-')
b

