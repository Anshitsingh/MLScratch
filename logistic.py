
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math 
from sklearn import datasets

train = pd.read_csv("~/Desktop/ASS1//train.csv")
test = pd.read_csv("~/Desktop/ASS1/test.csv")

data_train=pd.DataFrame(train,columns=['Pclass','Age','Fare','SibSp','Parch'])
data_train['Age'].fillna(data_train['Age'].mean(),inplace=True)


data_test=pd.DataFrame(test,columns=['Pclass','Age','Fare','SibSp','Parch'])
data_test['Age'].fillna(data_test['Age'].mean(),inplace=True)

target_train=pd.DataFrame(train,columns=['Survived'])
target_test=pd.DataFrame(test,columns=['Survived'])




# In[2]:


data_train.isnull().sum()


# In[3]:


data_test.isnull().sum()


# In[4]:


data_train.head()


# In[5]:


data_train['b_values']=1


# In[6]:


data_train.head()


# In[7]:


data_test['b_values']=1
data_test.head()


# In[8]:


def sigmoid(x):
    return(1.0/(1.0+np.exp(-1.0*x)) )


# In[9]:



weights=np.zeros((1,6))


# In[10]:


weights


# In[11]:


def stochastic_grad_descent_logistic(data,target,weights,alpha,numiterations):
    m=len(target)
    w_l=len(weights.T)
        
    for k in range(numiterations):
        f=weights.T
        p= data.dot(f)
        predicted=sigmoid(p)
        diff=np.subtract(target,predicted)
        
        loss=((-np.dot(target.T,(np.log(predicted))))-np.dot((1-target).T,(np.log(1-predicted))))/m
        
        
        
        delta=1e-1
        if(loss>delta):
            for i in range(m):            
                for j in range(w_l-1):
                    gradient=(data.iloc[i][j])*(np.subtract(target.iloc[i],predicted.iloc[i]))/m
                    weights[0][j]=weights[0][j]+alpha*gradient
                
                gradient_b=np.subtract(target.iloc[i],predicted.iloc[i])/m
                weights[0][w_l-1]=weights[0][w_l-1]+alpha*gradient_b
                    
                p=(data.dot(f))
                predicted=sigmoid(p)
                diff=np.subtract(target,predicted)
                loss=((-np.dot(target.T,(np.log(predicted))))-np.dot((1-target).T,(np.log(1-predicted))))/m
            print "loss after ",k,"th iteration : ",loss
            
        if(k>5):
            alpha=alpha/2        
            
        
        
    return weights


# In[12]:


def logistic_regression_fit(data_train,target_train):
    data=data_train
    target=target_train
    
    x=len(data.columns)
    weights=np.zeros((1,x))
    
    alpha=.002
    numiterations=10
    k=weights.T
    p=(data_train.dot(k))
    m=len(target_train)

    
    predicted=sigmoid(p)
    
    
    diff=np.subtract(target,predicted)
    
    loss=((-np.dot(target.T,(np.log(predicted))))-np.dot((1-target).T,(np.log(1-predicted))))/m
    
    
    
        
    
    weights=stochastic_grad_descent_logistic(data,target,weights,alpha,numiterations)
    p=(data_train.dot(weights.T))
    predicted=sigmoid(p)
    
    diff=np.subtract(target,predicted)
    
    loss=((-np.dot(target.T,(np.log(predicted))))-np.dot((1-target).T,(np.log(1-predicted))))/m
        
    
    return loss,weights


# In[20]:


def logistic_regression_predict(data_test,target_test,weights):
    p=(data_test.dot(np.transpose(weights)))
    m=len(target_test)
    predicted=sigmoid(p)
    target=target_test
    
    diff=np.subtract(target,predicted)
    
    loss=((-np.dot(target.T,(np.log(predicted))))-np.dot((1-target).T,(np.log(1-predicted))))/m
    threshold=.5    
    
    classified=np.zeros((1,len(target_test)) )
    for k in range(len(data_test)):
        if(predicted.iloc[k]>threshold):
            classified.iloc[0][k]=1
            if(target[k]!= classified[0][k]):
                incorrect=incorrect+1
        if(predicted.iloc[k]<threshold):
            classified[0][k]=0
            if(target[k]!=classified[0][k]):
                incorrect=incorrect+1

    accuracy=incorrect/len(target_test)
    return classified,accuracy


# In[14]:


loss,weights=logistic_regression_fit(data_train,target_train)


# In[21]:


classified,accuracy=logistic_regression_predict(data_test,target_test,weights)
print accuracy

