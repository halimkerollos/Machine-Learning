#!/usr/bin/env python
# coding: utf-8

# Using a small dataset on Kyphosis, I will create a decision trees and random foresets to predict whether or not the corrective spine surgery was successful.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("kyphosis.csv")
df.head()


# Kyphosis is a spinal condition that requires surgery.
# Column descriptions: under Kyphosis, we have values that tell us if the condition was present or absent after the operation.
# Age in months is the nect columns.
# Number is the number of vertebra involved in the operation.
# Start is the number of the first or top-most vertebra operated on.

# In[5]:


df.info()


# In[6]:


df.describe()


# In[8]:


sns.pairplot(df, hue="Kyphosis")


# In[9]:


# In[10]:


X = df.drop("Kyphosis", axis=1)
y = df["Kyphosis"]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[12]:


# In[13]:


dtree = DecisionTreeClassifier()


# In[14]:


dtree.fit(X_train, y_train)


# In[15]:


pred = dtree.predict(X_test)


# In[16]:


# In[17]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[18]:


# In[19]:


rfc = RandomForestClassifier(n_estimators=200)


# In[21]:


rfc.fit(X_train, y_train)


# In[23]:


rfc_pred = rfc.predict(X_test)


# In[24]:


print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))


# A bit better but still pretty low accuracy, precision, and recall.
# A reason for this could be that it's very small dataset and the train test split might have been unideal. The second potential issue is the heavy imbalance of labeled data, as I will demonstrate better in the next cell. But basically there are far more data points lebeled 'absent' than 'present' which combined with the small dataset and random state might have given us low valued metrics.

# In[33]:


df["Kyphosis"].value_counts()


# In[34]:


absent = df["Kyphosis"].value_counts()[0]
absent / sum(df["Kyphosis"].value_counts())


# So 79% of our data is labeled 'absent' which is a very poorly balanced dataset.
# But at the very least it looks like the random forests certainly performed better than a single decision tree and I am looking forward to applying this to larger, more complex datasets.

# Below I will show two different ways to plot our decision trees, so we have a little more visualization on the mathmatics.

# In[36]:


from IPython.display import Image
from io import StringIO
import pydot

features = list(df.columns[1:])
features


# In[48]:


dot_data = StringIO()
export_graphviz(
    dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True
)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())


# In[39]:


# In[47]:


plt.figure(figsize=(30, 20))
tree.plot_tree(dtree)
