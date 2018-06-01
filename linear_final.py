
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math 
from sklearn import datasets


d=pd.read_csv("~/Desktop/ASS1/winequality-red.csv",delimiter=';')
da=pd.DataFrame(d)
x=da.iloc[:,:-1]
data=pd.DataFrame(x)
q=da.iloc[:,-1:]
target=pd.DataFrame(q)

i=(len(da))

data.loc[:,'b_values']=np.ones((i,1))

m=pd.DataFrame(data)


# In[2]:


def train_test_split(data,target,train_fraction):
    x=int(math.ceil(train_fraction*len(target)))
    train_data=pd.DataFrame(data.iloc[:x][:])
    train_target=pd.DataFrame(target.iloc[:x][:])
    test_data=pd.DataFrame(data.iloc[x:][:])
    test_target=pd.DataFrame(target.iloc[x:][:])
    return train_data,train_target,test_data,test_target


# In[3]:


train_data,train_target,test_data,test_target=train_test_split(data,target,.8)
features=len(train_data.columns)
weights=np.ones((1,features))
features


# In[4]:


def stochastic_grad_descent_linear(data,target,weights,alpha,numiterations):
    m=len(target)
    w_l=len(weights.T)
        
    for k in range(numiterations):
        f=weights.T
        predicted=(data.dot(f))
        diff=np.subtract(target,predicted)
        mse=np.sum((diff)**2)/(2*m)
        delta=1e-1
        
        print mse
        
        if((mse>delta).bool()):
            for i in range(m):            
                for j in range(w_l-1):
                    gradient=(data.iloc[i][j])*(np.subtract(target.iloc[i],predicted.iloc[i]))/m
                    weights[0][j]=weights[0][j]+alpha*gradient
                
                gradient_b=np.subtract(target.iloc[i],predicted.iloc[i])/m
                weights[0][w_l-1]=weights[0][w_l-1]+alpha*gradient_b
                    
                predicted=(data.dot((weights).T))
                
                
                diff=np.subtract(target,predicted)
                mse=np.sum((diff)**2)/(2*m)
                print ("mse after one datapoint update:",mse)
        alpha=alpha/2
        
        
        
        
     
    return weights


# In[5]:


def linear_regression_fit(data_train,target_train):
    data=data_train
    target=target_train
    
    x=len(data.columns)
    weights=np.ones((1,x))
    
    alpha=.1
    numiterations=100
    m=weights.T
    predicted=(data_train.dot(m))
    
    
    weights=stochastic_grad_descent_linear(data,target,weights,alpha,numiterations)
    predicted=(data_train.dot(np.transpose(weights)))
    m=len(target)
    diff=np.subtract(target,predicted)
    mse=np.sum((diff)**2)/(2*m)
    return mse,weights


# In[6]:


def linear_regression_predict(data_test,target_test,weights):
    predicted=(data_test.dot(np.transpose(weights)))
    diff=np.subtract(target_test,predicted)
    mse=np.sum((diff)**2)/(2*m)
    return predicted,mse


# In[7]:



mse,weights=linear_regression_fit(train_data,train_target)


# In[9]:


pred,mse=linear_regression_predict(test_data,test_target,weights)
diff=np.subtract(pred,test_target)
mse=np.sum((diff)**2)
sse=np.sum((test_target-test_target.mean())**2)
print mse,sse

r_squared=1-mse/sse


