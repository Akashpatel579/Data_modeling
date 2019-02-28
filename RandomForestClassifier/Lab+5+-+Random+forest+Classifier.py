
# coding: utf-8

# In[113]:


#imort pandas
import pandas as pd


# In[114]:


#read dataset
dataset = pd.read_csv('SampData_DecisionTree.csv')


# In[115]:


dataset.head(2)


# In[116]:


print(dataset.shape)


# In[117]:


feature_col_names = dataset.columns[:-1]


# In[118]:


#features comlumn names
print(feature_col_names)


# In[119]:


# To convert text data to numeric data for random forest classifier
from sklearn.preprocessing import LabelEncoder
le = preprocessing.LabelEncoder()


# In[120]:


# apply this transforamtion to whole dataframe
dataset = dataset.apply(LabelEncoder().fit_transform)


# In[121]:


dataset.head(5)


# In[122]:


# prepare dataset for features X and y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,(dataset.shape[1]-1)].values


# In[123]:


print(y.shape)


# In[124]:


print (X.shape)


# In[125]:


# import train_test_split
from sklearn.model_selection import train_test_split


# In[126]:


# To check all the data columns in numric form
dataset.describe()


# In[127]:


# Split dataset into train and test dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.30, random_state=23)


# In[128]:


print('Training Features Shape:', train_X.shape)
print('Training Labels Shape:', train_y.shape)
print('Testing Features Shape:', test_X.shape)
print('Testing Labels Shape:', test_y.shape)


# In[129]:


# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier


# In[130]:


# Create a random forest Classifier
clf = RandomForestClassifier(n_jobs=2, random_state=23)

# Train the Classifier to take the training features train_X and learn how they relate to the training train_y 
clf.fit(train_X, train_y)


# In[131]:


clf.feature_importances_


# In[132]:


# Enable plot in jupyter notebook
get_ipython().magic('matplotlib inline')

#features importance plot
feat_importances = pd.Series(clf.feature_importances_, index=feature_col_names)
feat_importances.nlargest(10).plot(kind='barh')


# In[133]:


# prediction across the test_X data points
y_pred = clf.predict(test_X)


# In[134]:


# import cocnfusion_matrix 
from sklearn.metrics import confusion_matrix


# In[135]:


# Create confusion matrix
# 0 = "<=50k" , 1 = ">50k"
pd.crosstab(test_y, y_pred, rownames=['Actual income'], colnames=['Predicted income'])


# In[136]:


conf_mat = confusion_matrix(test_y, y_pred)
print(conf_mat)


# In[137]:


# True positive and Flase Negative
(conf_mat[0][0]+conf_mat[1,1] ) / conf_mat.sum()

