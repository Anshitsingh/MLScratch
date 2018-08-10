
# coding: utf-8

# In[1]:


import pandas as pd
a=pd.read_csv('/home/jash/Desktop/petplan/profit/ProfitabilityData_V2.csv', sep='|', error_bad_lines=False)


# In[2]:


len(a)


# In[3]:


a.head()


# In[4]:


a.isnull().sum()


# In[5]:


g=a.columns


# In[6]:


col=pd.DataFrame(index=range(len(a.columns)),columns=['Columns'])
print len(g)
i=0
for i in range(len(g)):
    
    col.set_value(i,'Columns',str(g[i]))
    
    


# In[7]:


col.to_csv('/home/jash/Desktop/petplan/profit/columns.csv')


# In[8]:


a['Mix'] = pd.np.where( a['BreedName'].str.contains("Mix"), 1, 0)


# In[9]:


len(a['TransactionNumber'].unique())


# In[10]:


len(a['AnnualDedInd'].unique())


# In[11]:


len(a['BreedName'].unique())


# In[12]:


len(a['PetId'].unique())


# In[13]:


len(a['Country'].unique())


# In[14]:


len(a['ControllingStateCd'].unique())


# In[15]:


len(a['ControllingStateCd'].unique())


# In[16]:


a['ControllingStateCd'].unique()


# In[17]:


len(a['ControllingStateCd'].unique())


# In[18]:


a.columns


# In[19]:


a.drop(['CancelDt','CrtdDateTime','CrtdUser','Effectivedt','EmailAddr','PetId','PetName','PolicyEndDate','PolicyDisplayNumber','PolicyRef','RenewedFromPolicyRef','UpdateTimestamp','TransactionNumber','EarnedPremiumAmtLocal','LastAnnualPremiumAmtLocal','InitialWrittenPremiumAmtLocal','TransactionDt','ExpirationDt','SourceSystem','PayplanCd','CampaignCd','Processed'],axis=1,inplace=True)


# In[20]:


a.drop(['TransactionCd','InitialWrittenPremiumAmt','LastAnnualPremiumAmt','BreedName'],axis=1,inplace=True)


# In[21]:


a.isnull().sum()


# In[22]:


m=a.loc[a['PolicyForm']=='Introductory']
len(m)


# In[23]:


a.isnull().sum()


# In[24]:


a.isnull().sum()


# In[25]:


a.isnull().sum()


# In[26]:


len(a['AnnualDedInd'].unique())


# In[27]:


a['LossRatio']=a['ClaimAmount']/a['EarnedPremiumAmt']


# In[28]:


k=len(a)


# In[29]:


import numpy as np
a = a[np.isfinite(a['LossRatio'])]


# In[30]:


u=len(a)
print u


# In[31]:



x=a[:10000]
d=pd.DataFrame(x)
d.to_csv('/home/jash/Desktop/petplan/profit/mod.xlsx')


# In[32]:


a.loc[a['CustomerNumber']=='EF41FAED66C44631910A92E41FAA7F78']


# In[33]:


len(a['CustomerNumber'].unique())


# In[34]:


r=a.loc[a['CustomerNumber'].duplicated()==True]
rep=pd.DataFrame(r)


# In[35]:


len(rep)


# In[36]:


rep.head()


# In[37]:


f=rep.loc[rep['CustomerNumber']=='41720']


# In[38]:


fe=pd.DataFrame(f)


# In[39]:


fe.to_csv('/home/jash/Desktop/petplan/profit/yo.xlsx')


# In[40]:


a.head()


# In[41]:


a.columns


# In[42]:


a.isnull().sum()


# In[43]:


a.drop(['StatusCd','ControllingStateCd'],axis=1,inplace=True)


# In[44]:


len(a)


# In[45]:


a.columns


# In[46]:


a['PetType'].unique()


# In[47]:


a['PetType'].replace({'PPCAT001': 'Cat','PPDOG001': 'Dog'},inplace=True)


# In[48]:


a['PetType'].unique()


# In[49]:


a['CopayPct'].replace({10:.9,20:.8,30:.7,0:1},inplace=True)


# In[50]:


a['CopayPct'].unique()


# In[51]:


len(a)


# In[52]:


a.columns


# In[53]:


new_index=np.array((range(len(a))))

a = a.reset_index(drop=True)
print a.head()


# In[54]:


a.head()


# In[55]:


a['CarrierCd'].unique()


# In[56]:


len(a['CarrierCd'].unique())


# In[57]:


a.loc[a['Country']=='CAN']['ExchangeRate']


# In[58]:


len(a['PolicyForm'].unique())


# In[59]:


len(a['AnnualDedInd'].unique())


# In[60]:


a.columns


# In[61]:


a.drop(['EarnedPremiumAmt','CustomerNumber','ExchangeRate'],axis=1,inplace=True)


# In[62]:


cols_to_transform = ['AnnualDedInd','CarrierCd','ControllingStateCd','Country','PetType','PolicyForm']
df = pd.get_dummies(d)


# In[63]:


df.head()

