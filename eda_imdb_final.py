
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a=pd.read_csv("~/Desktop/ASS1/movie_metadata.csv")
data=pd.DataFrame(a)

data.describe(include='all')


# In[52]:


data.columns


# In[3]:


lang=data.groupby(['language'],sort=False)
lang.describe()


# In[4]:


for name,group in lang:
    print name
    print group


# In[54]:


englishmovies=pd.DataFrame(lang.get_group('English'))
print englishmovies.columns


# In[106]:


dir=englishmovies.groupby(['director_name'],sort=False)[['director_facebook_likes']].sum()
a=pd.DataFrame(dir)

d=a.sort_values(by=['director_facebook_likes'],ascending=False)[:10][:]
print d


# In[120]:


plt.figure(figsize=(20,10))
plt.scatter(d.index,d['director_facebook_likes'])
plt.title('Most liked facebook directors')


# In[139]:


dir=englishmovies.groupby(['genres'],sort=False)[['gross']].sum()
p=pd.DataFrame(dir)

q=p.sort_values(by=['gross'],ascending=False)[:10][:]
print q



# In[141]:


q['gross']=q['gross']/(10**9)
q


# In[145]:


plt.figure(figsize=(20,10))
plt.title("Highest grossing genres in billions of dollars")
plt.scatter(q.index,q['gross'])
plt.ylim(0,10)
plt.show()


# In[174]:


correlation=englishmovies.corr(method='pearson')
correlation




# In[193]:



c=np.zeros((len(correlation),len(correlation.columns)))

for i in range(len(correlation)):
    for j in range(i+1,len(correlation.columns)):
        
        if((correlation.iloc[i][j]>0.5)or(correlation.iloc[i][j]<-.5)):
            c[i][j]=1
            print "strong correlation bw",correlation[i].index,"and",correlation[j].column,"\n"
        else:
            c[i][j]=0


# In[154]:


plt.figure(figsize=(20,10))
plt.scatter(englishmovies['gross'],englishmovies['imdb_score'])

plt.title('Trying to find if movies with high imdb scores are also highly grossing')
plt.show()


# In[158]:


plt.figure(figsize=(20,10))
plt.xlim(0,10)
plt.scatter(englishmovies['facenumber_in_poster'],englishmovies['cast_total_facebook_likes'],color='red')

plt.title('Trying to find if number of faces in poster have high correlation to fb likes')
plt.show()


# In[160]:


dir=englishmovies.groupby(['genres'],sort=False)[['imdb_score']].mean()
a=pd.DataFrame(dir)

d=a.sort_values(by=['imdb_score'],ascending=False)[:10][:]
print d


# In[170]:


plt.figure(figsize=(20,10))
plt.title("Highest average imdb scores and their genres")
plt.scatter(d.index,d['imdb_score'])
plt.ylim(0,10)
plt.show()


# In[173]:


plt.figure(figsize=(20,10))
englishmovies['budget']=englishmovies['budget']/(10**8)
plt.scatter(englishmovies['duration'],englishmovies['imdb_score'],color='yellow')
plt.scatter(englishmovies['budget'],englishmovies['imdb_score'],color='red')
plt.title('Duration and budget vs imdb_score')
plt.show()

