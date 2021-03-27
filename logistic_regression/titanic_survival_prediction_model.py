#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('titanic_train.csv')
train.info()


# How much data are we missing? 

# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Trends on survival?

# In[ ]:


sns.countplot(x = 'Survived', data = train, hue = 'Sex')


# In[ ]:


sns.countplot(x = 'Survived', data = train, hue = 'Pclass')


# Distribution of age on Titanic

# In[ ]:


sns.displot(train['Age'].dropna(), bins=30, kde=True)


# In[ ]:


train.columns


# In[ ]:


sns.countplot(x='SibSp', data = train)


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


train['Fare'].iplot(kind='hist', bins=50, )


# What we need to do now is deal with the null data, age is more pertinant to what we want to know so we'll deal with that first. One thing we can do is fill in all the null values with the average value (imputation). In other words we can calculate the average value of all passengers on titanic and stick that into all the rows where age is null. We can be even smarter about this by calculating the average values based on a more specific category such as class:

# In[ ]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Pclass', y='Age', data = train)


# In[ ]:


def imputeAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age']=train[['Age', 'Pclass']].apply(imputeAge, axis=1)


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Cool so now we've taken care of the age column and all of it's null values. The cabin column has way too many missing features to make any educated guesses on filling them, and the other option woud be to impute a binary value based on whether or not we knew the cabin(1 for some value, 0 for null), but I personally don't se how knowing this could help us in our goal, so I will be dropping the whole column.

# In[ ]:


train.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Great, now we have no missing values and can continue with our machiene learning classification problem.

# In[ ]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
sex.head()


# In[ ]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)
embark.head()


# In[ ]:


train = pd.concat([train,sex,embark], axis=1)
train.head()


# In[ ]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.head()


# In[ ]:


train.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


train.head()


# Now we have a perfect dataframe for a machine learning algorithm to do it's thing. One thing to consider is that Pclass is actually a categorical column and may have a different result if we used get_dummies instead of leaving it for the machine learning algorithm to interpret it as a continuous variable. I will seperate it after running my algorithm the first time in order to find out.

# In[ ]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression(max_iter=999999)


# In[ ]:


logmodel.fit(X_train, y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test, predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, predictions)


# So what this tells us is that our model is 79% accurate. Could be better and so I will be going back and tweak certain parameters and information that gets fed into the model for an attempt at a higher accuracy. Goal is to be above 90%
