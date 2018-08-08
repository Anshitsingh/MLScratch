
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
data.loc[data['LossRatio'] >.57 ,'L'] = 1
data.to_csv("/home/jash/Desktop/petplan/Profitability.csv")
Y=data['L']
y=pd.DataFrame(Y)
x=pd.DataFrame(data)

m=x.loc[x['PolicyForm']!='Introductory']
q=pd.DataFrame(m)
m2=q.loc[q['PolicyForm']!='Intro']
x=pd.DataFrame(m2)

print x.head()


# In[3]:


len(x)


# In[4]:


x.columns


# In[5]:


x.isnull().sum()


# In[6]:


x.drop(['LossRatio'],axis=1,inplace=True)
x.drop(['CustomerNumber','PhoneNumber','Surname','GivenName','CustomerMailingAddr_Addr1','CustomerMailingAddr_City','CustomerMailingAddr_StateProvCd',
       'CustomerMailingAddr_PostalCode','PetId','StartDate','EndDate','LastPolicyRef'],axis=1,inplace=True)


# In[7]:


p=pd.DataFrame(x)


# In[8]:


len(p['BreedName'].unique())


# In[9]:


p.isnull().sum()


# In[10]:


p.drop(['ClaimNumber','ClaimAmount','Severity','ClaimDetails','ConditionGrp','Claimcodecategory','claimdurationInception'],axis=1,inplace=True)


# In[11]:


p.isnull().sum()


# In[12]:


p.drop(['TotalClaimsAmtPaid'],axis=1,inplace=True)


# In[13]:


p.isnull().sum()


# In[14]:


len(p)


# In[15]:


p.dropna(how='any',inplace=True)


# In[16]:


p.drop(['churn\r'],axis=1,inplace=True)


# In[17]:


p.isnull().sum()


# In[18]:


len(p['BreedName'].unique())


# In[19]:


len(p['PetType'].unique())


# In[20]:


len(p['PolicyForm'].unique())


# In[21]:


len(p['PolicyForm'].unique())


# In[22]:


p.drop(['BreedName'],axis=1,inplace=True)


# In[23]:


p.isnull().sum()


# In[24]:


p.columns


# In[25]:


cols_to_transform = ['PetType','PolicyForm','Country','Quadrant']
df = pd.get_dummies(p)


# In[26]:


df.head()


# In[27]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(random_state=0)
y=df['L']
df.drop(['L'],axis=1,inplace=True)
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
from sklearn.metrics import roc_auc_score

param_grid = {'max_depth': np.arange(5, 10)}
from sklearn.model_selection import GridSearchCV
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
dtree.fit(X_train,y_train)


# In[28]:


dtree.get_params


# In[29]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[30]:


y_predict = dtree.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)


# In[32]:


from sklearn.metrics import confusion_matrix

pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted High Loss', 'Predicted Low Loss'],
    index=['True High Loss', 'True Low Loss']
)

