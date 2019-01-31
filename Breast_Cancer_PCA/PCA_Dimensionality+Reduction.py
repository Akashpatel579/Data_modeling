
# coding: utf-8

# In[19]:


# Import libraries 
from pandas import read_csv
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import


# In[20]:


#import dataset
dataframe = read_csv('breast-cancer-data.csv')


# In[21]:


#treat categorical values
dataset = pd.get_dummies(dataframe)


# In[22]:


#drop unwanted columns
dataset.drop(['id','diagnosis_B'],axis=1, inplace=True)
#split dataset 
y = dataset.diagnosis_M
X = dataset.ix[:,0:30]


# In[23]:


# PCA=2
pca = PCA(n_components=2)
pca.fit(X)
principalComponents = pca.transform(X)
principalDf = pd.DataFrame(data = principalComponents, 
                           columns = ['PCA1', 'PCA2'])
print(pca.explained_variance_ratio_)
finalDf = pd.concat([principalDf,y],axis=1)


# In[24]:


#plot PCA=2 
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['b', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['diagnosis_M'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
               , finalDf.loc[indicesToKeep, 'PCA2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[25]:


#PCA = 3
pca3 = PCA(n_components=3)
pca3.fit(X)
principalComponents3 = pca3.transform(X)
principalDf3 = pd.DataFrame(data = principalComponents3, 
                    columns = ['PCA1', 'PCA2','PCA3'])
print(pca3.explained_variance_ratio_)
finalDf3 = pd.concat([principalDf3,y],axis=1)


# In[26]:


#plot PCA=3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
targets = [0, 1]
colors = ['b', 'r']
ax.set_title('3 component PCA', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf3['diagnosis_M'] == target
    ax.scatter(finalDf3.loc[indicesToKeep, 'PCA1']
               , finalDf3.loc[indicesToKeep, 'PCA2']
               , finalDf3.loc[indicesToKeep, 'PCA3']
               , c = color
               , s = 50)

