
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


data.columns


# In[13]:


data.to_csv("/home/jash/Desktop/petplan/Profitability.csv")


# In[14]:


Y=data['L']


# In[15]:


y=pd.DataFrame(Y)


# In[16]:


y.head()


# In[17]:


x=pd.DataFrame(data)


# In[18]:


x.drop(['LossRatio'],axis=1,inplace=True)


# In[19]:


data['HighTenure-HighLR']=0
data['HighTenure-LowLR']=0
data['LowTenure-HighLR']=0
data['LowTenure-LowLR']=0


# In[20]:


data.columns


# In[21]:


data.loc[data['Quadrant']=='HighTenure-HighLR', 'HighTenure-HighLR'] = 1
data.loc[data['Quadrant']=='HighTenure-LowLR', 'HighTenure-LowLR'] = 1
data.loc[data['Quadrant']=='LowTenure-HighLR', 'LowTenure-HighLR'] = 1
data.loc[data['Quadrant']=='LowTenure-LowLR', 'LowTenure-LowLR'] = 1


# In[22]:


data.head()


# In[23]:


data.drop(['Quadrant'],axis=1,inplace=True)


# In[24]:


x.head()
x.to_csv("/home/jash/Desktop/petplan/xonly.csv")


# In[25]:


x.drop(['CustomerNumber','PhoneNumber','Surname','GivenName','CustomerMailingAddr_Addr1','CustomerMailingAddr_City','CustomerMailingAddr_StateProvCd',
       'CustomerMailingAddr_PostalCode','PetId','StartDate','EndDate','LastPolicyRef'],axis=1,inplace=True)


# In[26]:


x.columns


# In[27]:


x.head()


# In[28]:


len(x['BreedName'].unique())


# In[29]:


x.to_csv("/home/jash/Desktop/petplan/xonly_edited2.csv")


# In[30]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[31]:


m=.8*len(x)
m=340723
x_tr=x[:m]
y_tr=y[:m]
x_te=x[m:]
y_te=y[m:]


# In[35]:


x_tr.head()


# In[36]:


len(x['Severity'].unique())


# In[37]:


p=pd.DataFrame(x)


# In[38]:


p.columns


# In[39]:


p.drop(['BreedName','ClaimDetails','ConditionGrp'],axis=1,inplace=True)


# In[40]:


p.head()


# In[41]:


p['Quadrant'].unique()


# In[42]:


p['HighTenure-HighLR']=0
p['HighTenure-LowLR']=0
p['LowTenure-HighLR']=0
p['LowTenure-LowLR']=0
p.loc[p['Quadrant']=='HighTenure-HighLR', 'HighTenure-HighLR'] = 1
p.loc[p['Quadrant']=='HighTenure-LowLR', 'HighTenure-LowLR'] = 1
p.loc[p['Quadrant']=='LowTenure-HighLR', 'LowTenure-HighLR'] = 1
p.loc[p['Quadrant']=='LowTenure-LowLR', 'LowTenure-LowLR'] = 1


# In[43]:


p.drop(['Quadrant'],axis=1,inplace=True)


# In[44]:


p.columns


# In[45]:


p['Country'].unique()


# In[46]:


p['US']=0
p['CAN']=0
p.loc[p['Country']=='US', 'US'] = 1
p.loc[p['Country']=='CAN', 'CAN'] = 1
p.drop(['Country'],axis=1,inplace=True)


# In[47]:


p.columns


# In[48]:


p['PetType'].unique()


# In[49]:


p['Dog']=0
p['Cat']=0
p['PPDOG001']=0
p['PPCAT001']=0
p.loc[p['PetType']=='Dog', 'Dog'] = 1
p.loc[p['PetType']=='Cat', 'Cat'] = 1
p.loc[p['PetType']=='PPDOG001', 'PPDOG001'] = 1
p.loc[p['PetType']=='PPCAT001', 'PPCAT001'] = 1


# In[50]:


p.drop(['PetType'],axis=1,inplace=True)


# In[51]:


p.columns


# In[52]:


p['Claimcodecategory'].unique()


# In[53]:


p['Illness']=0
p['Accident']=0
p['Others']=0
p.loc[p['Claimcodecategory']=='Illness', 'Illness'] = 1
p.loc[p['Claimcodecategory']=='Accident', 'Accident'] = 1
p.loc[p['Claimcodecategory']=='Others', 'Others'] = 1


# In[54]:


p.drop(['Claimcodecategory','ClaimNumber'],axis=1,inplace=True)


# In[55]:


p.columns


# In[59]:


p['Severity'].unique()


# In[60]:


p['Curable']=0
p['Non-Curable']=0
p.loc[p['Severity']=='Curable', 'Curable'] = 1
p.loc[p['Severity']=='Accident', 'Non-Curable'] = 1



# In[ ]:


p.drop(['Severity'],axis=1,inplace=True)


# In[61]:


p.columns


# In[63]:


p.head()


# In[65]:


p.drop(['Severity'],axis=1,inplace=True)
p.columns


# In[66]:


p.head()


# In[67]:


p['CarrierCd'].unique()


# In[77]:


p.drop(p.index[334476],inplace=True)


# In[78]:


p['XLC']=0
p['ALZ']=0
p['CAN']=0
p['AGR']=0
p.loc[p['CarrierCd']=='XLC', 'XLC'] = 1
p.loc[p['CarrierCd']=='ALZ', 'ALZ'] = 1
p.loc[p['CarrierCd']=='CAN', 'CAN'] = 1
p.loc[p['CarrierCd']=='AGR', 'AGR'] = 1


# In[79]:


p.drop(['CarrierCd'],axis=1,inplace=True)
p.columns


# In[80]:


p.head()


# In[81]:


p['PolicyForm'].unique()


# In[82]:


p.drop(['PolicyForm'],axis=1,inplace=True)


# In[83]:


p.head()


# In[91]:


m=.8*len(p)
m=340720
print m
x_tr=p[:m]
y_tr=y[:m]
x_te=p[m:]
y_te=y[m:]


# In[92]:


p.head()


# In[93]:


import xgboost as xgb
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_tr, y_tr)
predictions = gbm.predict(x_te)


# In[ ]:


predictions==y_te


# In[ ]:


enc = preprocessing.OneHotEncoder()
enc.fit(X_2)
# 3. Transform
onehotlabels = enc.transform(X_2).toarray()
onehotlabels.shape


# In[ ]:


import h2o
from h2o.estimators import H2ORandomForestEstimator


# In[ ]:


# TODO: create a OneHotEncoder object, and fit it to all of X
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X_2 = x.apply(le.fit_transform)
X_2.head()




enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(x)

# 3. Transform
onehotlabels = enc.transform(X_2).toarray()
onehotlabels.shape

# as you can see, you've the same number of rows 891
# but now you've so many more columns due to how we changed all the categorical data into numerical data


# In[ ]:


import pandas as pd
k=pd.read_csv("/home/jash/Desktop/petplan/xonly_edited2.csv")
k.head()


# In[ ]:


model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)


# In[ ]:


h2o.init()
df = h2o.import_file(path="/home/jash/Desktop/petplan/xonly_edited2.csv")
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

