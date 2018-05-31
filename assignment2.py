import pandas as pd
import numpy as np
import math 
from sklearn import datasets


d=pd.read_csv("C:/Users/user/Desktop/assign/winequality-red.csv",delimiter=';')
da=pd.DataFrame(d)
x=da.iloc[:,:-1]
data=pd.DataFrame(x)
q=da.iloc[:,-1:]
target=pd.DataFrame(q)
i=(len(da))
data.loc[:]['b_values']=np.ones((i,1))
m=pd.DataFrame(data.iloc[:1280][:])

def sigmoid(x):
    return(1/(1+np.exp(-x)) )

def train_test_split(data,target,train_fraction):
    x=int(math.ceil(train_fraction*len(target)))
    train_data=pd.DataFrame(data.iloc[:x][:])
    train_target=pd.DataFrame(target.iloc[:x][:])
    test_data=pd.DataFrame(data.iloc[x:][:])
    test_target=pd.DataFrame(target.iloc[x:][:])
    return train_data,train_target,test_data,test_target
    

train_data,train_target,test_data,test_target=train_test_split(data,target,.8)

train_data,train_target,test_data,test_target=train_test_split(data,target,.8)
features=len(train_data.columns)
weights=np.ones((1,features))
features

def stochastic_grad_descent_logistic(data,target,weights,alpha,numiterations):
    m=len(target)
    w_l=len(weights.T)
        
    for k in range(numiterations):
        f=weights.T
        p=(data.dot(f))
        predicted=sigmoid(p)
        diff=np.subtract(target,predicted)
        loss=-target*(np.log(predicted) )-(1-target)*(np.log(1-predicted) )
        loss_total=np.sum(loss)
        delta=1e-1
        
        print loss
        
        if((loss_total>delta).bool()):
            for i in range(m):            
                for j in range(w_l-1):
                    gradient=(data.iloc[i][j])*(np.subtract(target.iloc[i],predicted.iloc[i]))/m
                    weights[0][j]=weights[0][j]-alpha*gradient
                
                gradient_b=np.subtract(target.iloc[i],predicted.iloc[i])/m
                weights[0][w_l-1]=weights[0][w_l-1]-alpha*gradient_b
                    
                predicted=(data.dot((weights).T))
                predicted=sigmoid(p)
                diff=np.subtract(target,predicted)
                loss=-target*(np.log(predicted) )-(1-target)*(np.log(1-predicted) )
                loss_total=np.sum(loss)
                print loss_total
        alpha=alpha/2
    return weights


def logistic_regression_fit(data_train,target_train):
    data=data_train
    target=target_train
    
    x=len(data.columns)
    weights=np.ones((1,x))
    
    alpha=.01
    numiterations=10
    m=weights.T
    p=(data_train.dot(m))
    predicted=sigmoid(p)
    diff=np.subtract(target,predicted)
    loss=-target*(np.log(predicted) )-(1-target)*(np.log(1-predicted) )
    loss_total=np.sum(loss)
    
    weights=stochastic_grad_descent_logistic(data,target,weights,alpha,numiterations)
    p=(data_train.dot(np.transpose(weights)))
    predicted=sigmoid(p)
    diff=np.subtract(target,predicted)
    loss=-target*(np.log(predicted) )-(1-target)*(np.log(1-predicted) )
    loss_total=np.sum(loss)
    m=len(target)
    return loss_total,weights





def logistic_regression_predict(data_test,target_test,weights):
    pred=(data_test.dot(np.transpose(weights)))
    predicted=sigmoid(p)
    diff=np.subtract(target,predicted)
    loss=-target*(np.log(predicted) )-(1-target)*(np.log(1-predicted) )
    loss_total=np.sum(loss)
    incorrect=0
    classified=np.zeros((1,len(target_test)) )
    for k in range(len(data_test)):
        if(predicted[k]>threshold):
            classified[0][k]=1
            if(target[k]!= classified[0][k]):
                incorrect=incorrect+1
        if(predicted[k]<threshold):
            classified[0][k]=0
            if(target[k]!=classified[0][k]):
                incorrect=incorrect+1

    accuracy=incorrect/len()
    return classified,accuracy



loss,weights=logistic_regression_fit(train_data,train_target)

pred,acc=logistic_regression_predict(test_data,test_target,weights)


   