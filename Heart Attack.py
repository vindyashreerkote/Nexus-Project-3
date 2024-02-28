#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve

import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("heart.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.info()


# Missing Value Analysis

# In[6]:


df.isnull().sum()


# Unique Value Analysis

# In[7]:


for i in list(df.columns):
    print("{} -- {}".format(i, df[i].value_counts().shape[0]))


# In[8]:


categorical_list = ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]


# In[9]:


df_categoric = df.loc[:, categorical_list]
for i in categorical_list:
    plt.figure()
    sns.countplot(x = i, data = df_categoric, hue = "output")
    plt.title(i)


# In[10]:


numeric_list = ["age", "trtbps","chol","thalachh","oldpeak","output"]


# In[11]:


df_numeric = df.loc[:, numeric_list]
sns.pairplot(df_numeric, hue = "output", diag_kind = "kde")
plt.show()


# In[12]:


scaler = StandardScaler()
scaler


# In[13]:


scaled_array = scaler.fit_transform(df[numeric_list[:-1]])


# In[14]:


scaled_array 


# In[15]:


df_dummy = pd.DataFrame(scaled_array, columns = numeric_list[:-1])
df_dummy.head()


# In[16]:


df_dummy = pd.concat([df_dummy, df.loc[:, "output"]], axis = 1)
df_dummy.head()


# In[17]:


data_melted = pd.melt(df_dummy, id_vars = "output", var_name = "features", value_name = "value")
data_melted.head(20)


# In[18]:


plt.figure()
sns.boxplot(x = "features", y = "value", hue = "output", data= data_melted)
plt.show()


# In[19]:


plt.figure()
sns.swarmplot(x = "features", y = "value", hue = "output", data= data_melted)
plt.show()


# In[20]:


sns.catplot(x = "exng", y = "age", hue = "output", col = "sex", kind = "swarm", data = df)
plt.show()


# In[21]:


plt.figure(figsize = (14,10))
sns.heatmap(df.corr(), annot = True, fmt = ".1f", linewidths = .7)
plt.show()


# In[22]:


numeric_list = ["age", "trtbps","chol","thalachh","oldpeak"]
df_numeric = df.loc[:, numeric_list]
df_numeric.head()


# In[23]:


df.describe()


# In[24]:


for i in numeric_list:
    
    Q1 = np.percentile(df.loc[:, i],25)
    Q3 = np.percentile(df.loc[:, i],75)
    
    IQR = Q3 - Q1
    
    print("Old shape: ", df.loc[:, i].shape)
    
    upper = np.where(df.loc[:, i] >= (Q3 +2.5*IQR))
    
    lower = np.where(df.loc[:, i] <= (Q1 - 2.5*IQR))
    
    print("{} -- {}".format(upper, lower))
    
    try:
        df.drop(upper[0], inplace = True)
    except: print("KeyError: {} not found in axis".format(upper[0]))
    
    try:
        df.drop(lower[0], inplace = True)
    except:  print("KeyError: {} not found in axis".format(lower[0]))
        
    print("New shape: ", df.shape)


# In[25]:


df1 = df.copy()


# In[26]:


df1 = pd.get_dummies(df1, columns = categorical_list[:-1], drop_first = True)
df1.head()


# In[27]:


X = df1.drop(["output"], axis = 1)
y = df1[["output"]]


# In[28]:


scaler = StandardScaler()
scaler


# In[29]:


X[numeric_list[:-1]] = scaler.fit_transform(X[numeric_list[:-1]])
X.head()


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_test: {}".format(y_test.shape))


# In[31]:


logreg = LogisticRegression()
logreg


# In[32]:


logreg.fit(X_train, y_train)


# In[33]:


y_pred_prob = logreg.predict_proba(X_test)
y_pred_prob


# In[34]:


y_pred = np.argmax(y_pred_prob, axis = 1)
y_pred


# In[35]:


print("Test accuracy: {}".format(accuracy_score(y_pred, y_test)))


# In[36]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])


# In[37]:


plt.plot([0,1],[0,1],"k--")
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()


# In[38]:


lr = LogisticRegression()
lr


# In[39]:


penalty = ["l1", "l2"]

parameters = {"penalty":penalty}


# In[40]:


lr_searcher = GridSearchCV(lr, parameters)


# In[41]:


lr_searcher.fit(X_train, y_train)


# In[42]:


print("Best parameters: ",lr_searcher.best_params_)


# In[43]:


y_pred = lr_searcher.predict(X_test)


# In[44]:


print("Test accuracy: {}".format(accuracy_score(y_pred, y_test)))


# In[ ]:




