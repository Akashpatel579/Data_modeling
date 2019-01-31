
# coding: utf-8

# In[129]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import numpy as np

dataset = pd.read_csv('voice.csv')


# In[130]:


print(dataset.shape)


# In[131]:


print(dataset.columns)


# In[132]:


# Create dummies data | Data processing 

dataset = pd.concat([dataset,pd.get_dummies(dataset.label, prefix='label')],axis=1)

dataset.drop(['label', 'label_female'],axis=1, inplace=True)
print (dataset)


# In[133]:


dataset['label_male'].value_counts()


# In[134]:


#slice the dataset get all columns, except the last one (-1), to X and Y

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,(dataset.shape[1]-1)].values

print(X.shape)
print(y.shape)


# In[135]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[136]:


y_pred=logreg.predict(X_test)


# In[137]:


print(y_pred)


# In[138]:


# confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[139]:


def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrixprint_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14) figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# In[140]:


class_names = ['Male', 'Female']


# In[141]:


print_confusion_matrix(cnf_matrix, class_names)


# In[142]:


# Evaluating Model parameters
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[143]:


#variable importance using corelation matrix

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

corr = dataset.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[144]:


# improve accuracy and remove unnecessary variable 
dataset.drop(['skew', 'modindx','minfun'],axis=1, inplace=True)
print (dataset)


# In[145]:


#slice the dataset get all columns, except the last one (-1), to X. get column 4 to Y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,(dataset.shape[1]-1)].values

print(X.shape)
print(y.shape)


# In[146]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[147]:


y_pred=logreg.predict(X_test)


# In[148]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[149]:


print_confusion_matrix(cnf_matrix, class_names)


# In[150]:


# Evaluating Model again parameters
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[ ]:


#which depict accuracy has been improved : ~ 0.03%

