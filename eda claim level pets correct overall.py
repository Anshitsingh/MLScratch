
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

m=x.loc[x['PolicyForm']!='Introductory']
q=pd.DataFrame(m)
m2=q.loc[q['PolicyForm']!='Intro']
x=pd.DataFrame(m2)
print x['PolicyForm'].unique()


# In[42]:


x['Quadrant'].unique()


# In[3]:


len(x)


# In[4]:


x = x[pd.notnull(x['BreedName'])]


# In[5]:


x.isnull().sum()


# In[6]:


x['Duration'].min()


# In[7]:


#QUADRANT
q = x.groupby(['Quadrant']).mean()
quad=pd.DataFrame(q)
topquad=quad['LossRatio'].sort_values(ascending=False)
topquad


# In[8]:


x['Dseg']=0


# In[9]:


x.loc[x['Duration'] <100, 'Dseg'] = 1
x.loc[(x['Duration']>=100) & (x['Duration'] <200),'Dseg']=2
x.loc[(x['Duration']>=200) & (x['Duration'] <300),'Dseg']=3
x.loc[(x['Duration']>=300) & (x['Duration'] <400),'Dseg']=4
x.loc[(x['Duration']>=400) & (x['Duration'] <500),'Dseg']=5
x.loc[(x['Duration']>=500) & (x['Duration'] <600),'Dseg']=6
x.loc[(x['Duration']>=600) & (x['Duration'] <700),'Dseg']=7
x.loc[(x['Duration']>=700) & (x['Duration'] <800),'Dseg']=8
x.loc[(x['Duration']>=800) & (x['Duration'] <900),'Dseg']=9
x.loc[(x['Duration']>=900) ,'Dseg']=10


# In[10]:


#OVERALL
x['Dseg'].unique()


# In[11]:


#DURATION
dur = x.groupby(['Dseg']).mean()
dse=pd.DataFrame(dur)
dse.head()
import matplotlib.pyplot as plt
fig = plt.figure("LossRatio vs DurationBucket",figsize=(16,8))

plt.plot(dse.index,dse['LossRatio'])


# In[12]:


x.loc[x['LossRatio']>=x['LossRatio'].max()-400]['Duration'].mean()


# In[13]:


len(x.loc[x['LossRatio']>=x['LossRatio'].max()-500])


# In[14]:


len(x.loc[x['Duration'] <100])


# In[15]:


x['Duration'].mean()


# In[16]:


topdse=dse['LossRatio'].sort_values(ascending=False)
print topdse


# In[17]:


#BREEDS
df_agg = x.groupby(['BreedName']).mean()
breed=pd.DataFrame(df_agg)
breed.head()
topbreeds=breed['LossRatio'].sort_values(ascending=False)
topbreeds.head(20)


# In[31]:


#CURRENTAGE
age = x.groupby(['Currentage']).mean()
ag=pd.DataFrame(age)
ag.head()
topcurrentage=ag['LossRatio'].sort_values(ascending=False)
topcurrentage.head(10)


# In[41]:


import matplotlib.pyplot as plt
fig = plt.figure("LossRatio vs Currentage",figsize=(16,8))
print age.columns
plt.plot(ag.index,ag['LossRatio'])


# In[19]:


#POLICYFORM
p = x.groupby(['PolicyForm']).mean()
pf=pd.DataFrame(p)
toppf=pf['LossRatio'].sort_values(ascending=False)
toppf


# In[20]:


#CHURN
c = x.groupby(['churn\r']).mean()
ch=pd.DataFrame(c)
churn=ch['LossRatio'].sort_values(ascending=False)
churn


# In[21]:


#DEDUCTIBLE
d=x.groupby(['Deductible']).mean()
ded=pd.DataFrame(d)
ded.head()
topded=ded['LossRatio'].sort_values(ascending=False)
topded.head(10)

