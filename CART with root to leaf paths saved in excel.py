
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
data.to_csv("/home/jash/Desktop/petplan/Profitability.csv")
Y=data['LossRatio']
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


x.drop(['CustomerNumber','PhoneNumber','Surname','GivenName','CustomerMailingAddr_Addr1','CustomerMailingAddr_City','CustomerMailingAddr_StateProvCd',
       'CustomerMailingAddr_PostalCode','PetId','StartDate','EndDate','LastPolicyRef'],axis=1,inplace=True)


# In[7]:


p=pd.DataFrame(x)


# In[8]:


len(p['BreedName'].unique())


# In[9]:


g=p.loc[p['churn\r']==1]


# In[10]:


len(g)


# In[11]:


p.isnull().sum()


# In[12]:


p.drop(['ClaimNumber','ClaimAmount','Severity','ClaimDetails','ConditionGrp','Claimcodecategory','claimdurationInception'],axis=1,inplace=True)


# In[13]:


p.isnull().sum()


# In[14]:


p.drop(['TotalClaimsAmtPaid'],axis=1,inplace=True)


# In[15]:


p.isnull().sum()


# In[16]:


len(p)


# In[17]:


p.dropna(how='any',inplace=True)


# In[18]:


p.drop(['churn\r'],axis=1,inplace=True)


# In[19]:


p.isnull().sum()


# In[20]:


len(p['BreedName'].unique())


# In[21]:


len(p['PetType'].unique())


# In[22]:


len(p['PolicyForm'].unique())


# In[23]:


len(p['PolicyForm'].unique())


# In[24]:


p.drop(['BreedName'],axis=1,inplace=True)


# In[25]:


p.isnull().sum()


# In[26]:


p.columns


# In[27]:


cols_to_transform = ['PetType','PolicyForm','Country','Quadrant']
df = pd.get_dummies(p)


# In[28]:


df.head()


# In[29]:


df.columns.unique()


# In[30]:


from sklearn.tree import DecisionTreeRegressor
ktree=DecisionTreeRegressor(random_state=0,max_depth=6,max_leaf_nodes=10)
y=df['LossRatio']
df.drop(['LossRatio'],axis=1,inplace=True)
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
from sklearn.metrics import roc_auc_score


parameters={'max_depth': range(1,6,1),'min_samples_leaf':range(800,1000,100)}
clf_tree=DecisionTreeRegressor()
from sklearn.model_selection import GridSearchCV
dtree=GridSearchCV(clf_tree,parameters)



dtree.fit(X_train,y_train)


# In[31]:


dtree.best_estimator_.get_params


# In[32]:


y_predict = dtree.best_estimator_.predict(X_test)


# In[33]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict)


# In[34]:


y_trainpredict = dtree.best_estimator_.predict(X_train)
mean_squared_error(y_train, y_trainpredict)


# In[35]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(dtree.best_estimator_, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[36]:


dtree.best_estimator_.tree_


# In[37]:


def tree_to_code(tree, feature_names, Y):
    tree_ = dtree.best_estimator_.tree_
    feature_name = [
        feature_names[i] 
        for i in tree_.feature
    ]
    pathto=dict()

    global k
    k = 0
    def recurse(node, depth, parent):
        global k
        indent = "  " * depth

        if tree_.feature[node] != tree_.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s= "{} <= {} ".format( name, threshold, node )
            if node == 0:
                pathto[node]=s
            else:
                pathto[node]=pathto[parent]+' & ' +s

            recurse(tree_.children_left[node], depth + 1, node)
            s="{} > {}".format( name, threshold)
            if node == 0:
                pathto[node]=s
            else:
                pathto[node]=pathto[parent]+' & ' +s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k=k+1
            print(k,')',pathto[parent], tree_.value[node])
    recurse(0, 1, 0)


# In[38]:


dtree.best_estimator_.tree_.__getstate__()['nodes']


# In[39]:


def get_rules(dtc, df):
    rules_list = []
    values_path = []
    values = dtc.tree_.value

    def RevTraverseTree(tree, node, rules, pathValues):
        '''
        Traverase an skl decision tree from a node (presumably a leaf node)
        up to the top, building the decision rules. The rules should be
        input as an empty list, which will be modified in place. The result
        is a nested list of tuples: (feature, direction (left=-1), threshold).  
        The "tree" is a nested list of simplified tree attributes:
        [split feature, split threshold, left node, right node]
        '''
        # now find the node as either a left or right child of something
        # first try to find it as a left node            

        try:
            prevnode = tree[2].index(node)           
            leftright = '<='
            pathValues.append(values[prevnode])
        except ValueError:
            # failed, so find it as a right node - if this also causes an exception, something's really f'd up
            prevnode = tree[3].index(node)
            leftright = '>'
            pathValues.append(values[prevnode])

        # now let's get the rule that caused prevnode to -> node
        p1 = df.columns[tree[0][prevnode]]    
        p2 = tree[1][prevnode]    
        rules.append(str(p1) + ' ' + leftright + ' ' + str(p2))

        # if we've not yet reached the top, go up the tree one more step
        if prevnode != 0:
            RevTraverseTree(tree, prevnode, rules, pathValues)

    # get the nodes which are leaves
    leaves = dtc.tree_.children_left == -1
    leaves = np.arange(0,dtc.tree_.node_count)[leaves]

    # build a simpler tree as a nested list: [split feature, split threshold, left node, right node]
    thistree = [dtc.tree_.feature.tolist()]
    thistree.append(dtc.tree_.threshold.tolist())
    thistree.append(dtc.tree_.children_left.tolist())
    thistree.append(dtc.tree_.children_right.tolist())

    # get the decision rules for each leaf node & apply them
    for (ind,nod) in enumerate(leaves):

        # get the decision rules
        rules = []
        pathValues = []
        RevTraverseTree(thistree, nod, rules, pathValues)

        pathValues.insert(0, values[nod])      
        pathValues = list(reversed(pathValues))

        rules = list(reversed(rules))

        rules_list.append(rules)
        values_path.append(pathValues)
        
    return (rules_list, values_path)


# In[40]:


a,b=get_rules(dtree.best_estimator_, df)


# In[41]:


a[0][0]


# In[42]:


len(b)


# In[43]:


print a[0][0].split(" ",1)[1] 


# In[44]:


obs=pd.DataFrame(index=range(len(a)),columns=df.columns)


# In[45]:


obs.columns


# In[46]:


obs['Value']=np.zeros(len(obs))


# In[47]:


a[0][0].split(" ",1)[0]


# In[56]:


a[0][0].split(" ",1)[1]


# In[48]:


b[20][-1][0][0]


# In[49]:


obs.loc[5][a[0][0].split(" ",1)[0]]


# In[50]:


obs.columns


# In[70]:


for i in range(len(a)):
    for k in range(len(a[i])):
        obs.set_value(i,a[i][k].split(" ",1)[0],a[i][k].split(" ",1)[1]) 
    obs.set_value(i,'Value',b[i][-1][0][0])
obs.dropna(axis=1,how='all',inplace=True)



# In[71]:


obs


# In[72]:


obs.to_csv("/home/jash/Desktop/petplan/CART/observations.xlsx")

