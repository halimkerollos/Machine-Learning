#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


cancer = load_breast_cancer()


# In[4]:


cancer.keys()


# In[5]:


print(cancer['DESCR'])


# In[6]:


print(cancer['feature_names'])


# In[7]:


df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()


# In[8]:


col = ['target']
target = pd.DataFrame(cancer['target'], columns=col)
target.info()


# In[9]:


cdf = df.join(target)
cdf


# In[10]:


X = cdf.drop('target', axis=1)
y = cdf['target']


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X_train)
strain = scaler.transform(X_train)
strain


# In[16]:


stest = scaler.transform(X_test)
stest


# In[17]:


strain.shape


# In[18]:


stest.shape


# In[19]:


from sklearn.decomposition import PCA


# In[20]:


pca = PCA(n_components=2)


# In[21]:


pca.fit(strain)


# In[22]:


strain_pca = pca.transform(strain)


# In[23]:


stest_pca = pca.transform(stest)


# In[24]:


strain_pca.shape


# In[25]:


stest_pca.shape


# In[26]:


plt.figure(figsize=(10, 4))
plt.scatter(strain_pca[:,1], strain_pca[:,0], c=y_train)
plt.xlabel('First Principle Component')
plt.ylabel('Second Principal Component')


# In[27]:


pca.components_


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


logR = LogisticRegression()


# In[30]:


logR.fit(strain_pca, y_train)


# In[31]:


pred = logR.predict(stest_pca)


# In[32]:


from sklearn.metrics import confusion_matrix, classification_report


# In[33]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[ ]:




