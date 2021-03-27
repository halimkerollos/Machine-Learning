#!/usr/bin/env python
# coding: utf-8

# Disclaimer: all customer information is fake, and purely for pracice and display of data analysis, visualization, and implemenation of machine learning.

# ## Imports
# ** Import pandas, numpy, matplotlib,and seaborn. Then set %matplotlib inline 
# (You'll import sklearn as you need it.)**

# In[1]:


import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[3]:


df = pd.read_csv('Ecommerce Customers')
df.head()


# **Check the head of customers, and check out its info() and describe() methods.**

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


sns.pairplot(df)


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[7]:


sns.jointplot(x = 'Yearly Amount Spent', y = 'Time on Website', data = df)


# ** Do the same but with the Time on App column instead. **

# In[8]:


sns.jointplot(x = 'Yearly Amount Spent', y = 'Time on App', data = df)


# In[9]:


sns.lmplot(x = 'Yearly Amount Spent', y = 'Time on App', data = df)


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[10]:


sns.jointplot(x = 'Length of Membership', y = 'Time on App', data = df, kind='hex')


# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# Definitely Membership Length seems to be the most correlated with Yearly Amount Spent.

# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[11]:


sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = df)


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[12]:


df.head()


# In[13]:


df.columns


# In[14]:


X=df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[17]:


from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

# In[18]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[19]:


lm.fit(X_train, y_train)


# **Print out the coefficients of the model**

# In[20]:


lm.coef_


# In[21]:


lm.intercept_


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[22]:


pred = lm.predict(X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[23]:


plt.scatter(y_test, pred)


# A pretty darn good line, great prediction on the part of the machine learning model.

# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[24]:


from sklearn import metrics


# In[25]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[26]:


metrics.explained_variance_score(y_test, pred)


# Explained Variance Score, above, describes the percentage of data that our model accurately predicts and in this case its just about 99%, which is pretty excellent.

# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[27]:


sns.displot(y_test-pred, bins=50, kde=True)


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# ** Recreate the dataframe below. **

# In[28]:


X_train.columns


# In[29]:


lm.coef_


# In[30]:


cdf=pd.DataFrame(lm.coef_, X_train.columns, columns = ['Coeff'])
cdf


# ** How can you interpret these coefficients? **

# From what it looks like we have the highest coefficient when it comes to Length of Membership, combined with the information from the pairplots above, I would say that is where the company should focus most of its efforts. Beyond that it appears that Time on App is the next greatest coefficient so they can also put their efforts into leveraging the app and it's features in order to maximize profits from the platform that is the most popular.

# **Do you think the company should focus more on their mobile app or on their website?**

# Well if they want a well balanced usage the website is suffering tremendously, so either completely redo that or just focus more on the app which seems to be far more popular and appears to generate much more income than the website.
# People like the app! Focus on the app.
