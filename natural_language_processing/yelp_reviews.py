#!/usr/bin/env python
# coding: utf-8

# In this project I will attempt to classify Yelp reviews into star categories beased on the text content in the review itself. I'll be doing this using Natural Language Processing techniques in python.
# The dataset: https://www.kaggle.com/c/yelp-recsys-2013

# In[1]:


from pyforest import *


# In[2]:


yelp = pd.read_csv('yelp.csv')
yelp.head()


# In[3]:


yelp.info()


# In[4]:


yelp.describe()


# In[15]:


yelp['txt_len'] = yelp['text'].apply(len)


# In[22]:


g = sns.FacetGrid(data = yelp, col='stars')
g.map(plt.hist, 'txt_len', bins=30, ec='black')


# In[24]:


sns.boxplot(x='stars', y='txt_len', data = yelp)


# In[30]:


sns.countplot(x = 'stars', data = yelp)


# In[41]:


stars =yelp.groupby('stars').mean()


# In[42]:


stars


# In[43]:


stars.corr()


# In[48]:


sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)


# To make this easier to start, I will only take the reviews that are either 1 star or 5 star.

# In[76]:


yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[78]:


X = yelp_class['text']
y = yelp_class['stars']


# In[79]:


from sklearn.feature_extraction.text import CountVectorizer


# In[80]:


countV = CountVectorizer()


# In[81]:


X = countV.fit_transform(X)


# In[84]:


X.shape


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[88]:


from sklearn.naive_bayes import MultinomialNB


# In[89]:


nb = MultinomialNB()


# In[90]:


nb.fit(X_train, y_train)


# In[91]:


pred = nb.predict(X_test)


# In[92]:


from sklearn.metrics import confusion_matrix, classification_report


# In[94]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[95]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[96]:


from sklearn.pipeline import Pipeline


# In[109]:


from sklearn.ensemble import RandomForestClassifier


# In[116]:


from sklearn.linear_model import LogisticRegression


# In[123]:


pipeline = Pipeline([('bow', CountVectorizer()),
                    ('bayes', LogisticRegression(max_iter=999999))
                    ])


# In[125]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[126]:


pipeline.fit(X_train, y_train)


# In[127]:


pred = pipeline.predict(X_test)


# In[128]:


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# Looks like we were able to make some improvements with logistic regression
