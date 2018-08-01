
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


x = x[pd.notnull(x['BreedName'])]


# In[4]:


x.isnull().sum()


# In[5]:


x['Duration'].min()


# In[13]:


#QUADRANT
q = x.groupby(['Quadrant']).mean()
quad=pd.DataFrame(q)
topquad=quad['LossRatio'].sort_values(ascending=False)
topquad


# In[35]:


x['Dseg']=0


# In[44]:


x.loc[x['Duration'] <200, 'Dseg'] = 1
x.loc[(x['Duration']>200) & (x['Duration'] <400),'Dseg']=2
x.loc[(x['Duration']>400) & (x['Duration'] <600),'Dseg']=3
x.loc[(x['Duration']>600) & (x['Duration'] <800),'Dseg']=4
x.loc[(x['Duration']>800 ),'Dseg']=5


# In[45]:


#OVERALL
x['Dseg'].unique()


# In[46]:


#DURATION
dur = x.groupby(['Dseg']).mean()
dse=pd.DataFrame(dur)
dse.head()
topdse=dse['LossRatio'].sort_values(ascending=False)
topdse


# In[10]:


#BREEDS
df_agg = x.groupby(['BreedName']).mean()
breed=pd.DataFrame(df_agg)
breed.head()
topbreeds=breed['LossRatio'].sort_values(ascending=False)
topbreeds.head(20)


# In[11]:


#POLICYFORM
p = x.groupby(['PolicyForm']).mean()
pf=pd.DataFrame(p)
toppf=pf['LossRatio'].sort_values(ascending=False)
toppf


# In[48]:


#CHURN
c = x.groupby(['churn\r']).mean()
ch=pd.DataFrame(c)
churn=ch['LossRatio'].sort_values(ascending=False)
churn


# In[12]:


#DEDUCTIBLE
d=x.groupby(['Deductible']).mean()
ded=pd.DataFrame(d)
ded.head()
topded=ded['LossRatio'].sort_values(ascending=False)
topded.head(10)


# In[14]:


#LowTenure-HighLR
m=x.loc[x['Quadrant']=='LowTenure-HighLR']
m=pd.DataFrame(m)
df_agg = m.groupby(['BreedName']).mean()
breed=pd.DataFrame(df_agg)
topbreeds=breed['LossRatio'].sort_values(ascending=False)
topbreeds.head(10)


# In[15]:


#POLICYFORM
p = m.groupby(['PolicyForm']).mean()
pf=pd.DataFrame(p)
toppf=pf['LossRatio'].sort_values(ascending=False)
toppf


# In[16]:


#DEDUCTIBLE
d=m.groupby(['Deductible']).mean()
ded=pd.DataFrame(d)
ded.head()
topded=ded['LossRatio'].sort_values(ascending=False)
topded.head(10)


# In[17]:


#HighTenure-HighLR
m=x.loc[x['Quadrant']=='HighTenure-HighLR']
m2=pd.DataFrame(m)
df_agg = m2.groupby(['BreedName']).mean()
breed=pd.DataFrame(df_agg)
topbreeds=breed['LossRatio'].sort_values(ascending=False)
topbreeds.head(10)


# In[18]:


#POLICYFORM
p = m2.groupby(['PolicyForm']).mean()
pf=pd.DataFrame(p)
toppf=pf['LossRatio'].sort_values(ascending=False)
toppf


# In[19]:


#DEDUCTIBLE
d=m2.groupby(['Deductible']).mean()
ded=pd.DataFrame(d)
ded.head()
topded=ded['LossRatio'].sort_values(ascending=False)
topded.head(10)


# In[20]:


#HighTenure-LowLR
m=x.loc[x['Quadrant']=='HighTenure-LowLR']
m3=pd.DataFrame(m)
df_agg = m3.groupby(['BreedName']).mean()
breed=pd.DataFrame(df_agg)
topbreeds=breed['LossRatio'].sort_values(ascending=False)
topbreeds.head(10)


# In[21]:


#POLICYFORM
p = m3.groupby(['PolicyForm']).mean()
pf=pd.DataFrame(p)
toppf=pf['LossRatio'].sort_values(ascending=False)
toppf


# In[22]:


#DEDUCTIBLE
d=m3.groupby(['Deductible']).mean()
ded=pd.DataFrame(d)
ded.head()
topded=ded['LossRatio'].sort_values(ascending=False)
topded.head(10)

