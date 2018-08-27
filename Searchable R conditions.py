
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
a=pd.read_csv("E:/Petplan/rules.csv")
b=[]
for i in range(len(a)):
    x=''
    for j in range(1,len(a.columns)):
        if (((str(a.ix[i][j]))!='nan')and (str(a.ix[i][j])!='when')):
            x=x+str(a.ix[i][j]).replace(' ','')
    b.append(x)

a=[]
for i in range(len(b)):
    a.append(np.asarray(b[i].split('&')))
a=np.asarray(a)

for i in range(len(a)):
    for k in range(len(a[i])):
        a[i][k].replace(' ','')

b=np.array(len(a))
conditions=[]
for i in range(len(a)):
    f='('
    for k in range(len(a[i])):
        m=a[i][k].split('is')
        m[0]='df$'+str(m[0])
        if(len(m)>1):
            t='('
            k=m[1].split('or')
            if(len(k)>1):
                t=t+m[0]+'%in% c( '
                for p in range(len(k)):
                    if(p!=len(k)-1):
                        t=t+'\''+str(k[p])+'\''+','
                    else:
                        t=t+'\''+str(k[p])+'\''+')'+')'
            else:
                t=t+m[0]+'==\''+str(m[1]) +'\')'
            m[1]=t
            f=f+t+'&'
        else:
            f=f+'('+m[0]+')'+'&'
    f=f[:-1]
    f=f+')'
    conditions.append(f)
    
conditions=np.asarray(conditions)
print conditions
condition=pd.DataFrame(conditions)
condition.to_csv("E:/Petplan/conditions.csv")    
    

