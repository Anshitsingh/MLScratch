
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math 
from sklearn import datasets

train = pd.read_csv("~/Desktop/ASS1//train.csv")

data=pd.DataFrame(train,columns=['Pclass','Age','Fare','SibSp','Parch','Survived'])
data['Age'].fillna(data['Age'].mean(),inplace=True)
target=pd.DataFrame(train,columns=['Survived'])

def train_test_split(data,target,train_fraction):
    x=int(math.ceil(train_fraction*len(target)))
    train_data=pd.DataFrame(data.iloc[:x][:])
    train_target=pd.DataFrame(target.iloc[:x][:])
    test_data=pd.DataFrame(data.iloc[x:][:])
    test_target=pd.DataFrame(target.iloc[x:][:])
    return train_data,train_target,test_data,test_target


data_train,target_train,data_test,target_test=train_test_split(data,target,.8)
features=len(data_train.columns)
weights=np.ones((1,features))
features





# In[2]:


data_train.isnull().sum()


# In[3]:


target_train.isnull().sum()


# In[4]:


target_test.head()


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


# In[91]:


def logistic_regression_predict(data_test,target_test,weights):
    p=(data_test.dot(np.transpose(weights)))
    m=len(target_test)
    predicted=sigmoid(p)
    target=target_test
    
    diff=np.subtract(target,predicted)
    
    loss=((-np.dot(target.T,(np.log(predicted))))-np.dot((1-target).T,(np.log(1-predicted))))/m
    threshold=.5    
   
    incorrect=0
    classified=np.zeros((1,len(target_test)) )
    for k in range(len(data_test)):
        a=predicted.iloc[k][0]
        threshold=.5  
        if(a>threshold):
            classified[0][k]=1
            if(target.loc[713+k][0]!= classified[0][k]):
                incorrect=incorrect+1
        if(a<threshold):
            classified[0][k]=0
            if(target.loc[713+k][0]!=classified[0][k]):
                incorrect=incorrect+1
    denominator=len(target_test)
    print "incorrect:\n",incorrect,"\ntotal_test_set_size:\n",denominator
    
    accuracy=float(float(denominator-incorrect)/float(denominator) )
    
    return classified,accuracy


# In[37]:


loss,weights=logistic_regression_fit(data_train,target_train)


# In[92]:



classified,accuracy=logistic_regression_predict(data_test,target_test,weights)
print "Accuracy of test set is : ",accuracy

