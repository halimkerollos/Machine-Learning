import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

cf.go_offline()
get_ipython().run_line_magic("matplotlib", "inline")

train = pd.read_csv("titanic_train.csv")
train.info()


# How much data are we missing?
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# Trends on survival?
sns.countplot(x="Survived", data=train, hue="Sex")
sns.countplot(x="Survived", data=train, hue="Pclass")


# Distribution of age on Titanic
sns.displot(train["Age"].dropna(), bins=30, kde=True)
train.columns
sns.countplot(x="SibSp", data=train)

train["Fare"].iplot(
    kind="hist",
    bins=50,
)


# What we need to do now is deal with the null data,
# age is more pertinant to what we want to know
# so we'll deal with that first.
# One thing we can do is fill in all the null values with the
# average value (imputation). In other words we can calculate the average
# value of all passengers on titanic and stick that into all the
# rows where age is null. We can be even smarter about this by
# calculating the average values based on a more specific category such
# as class:
plt.figure(figsize=(10, 5))
sns.boxplot(x="Pclass", y="Age", data=train)


def imputeAge(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train["Age"] = train[["Age", "Pclass"]].apply(imputeAge, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# Cool so now we've taken care of the age column and all of it's null values.
# The cabin column has way too many missing features to make
# any educated guesses on filling them, and the other option
# would be to impute a binary value based on whether or
# not we knew the cabin(1 for some value, 0 for null),
# but I personally don't se how knowing this could help us in our goal,
# so I will be dropping the whole column.

train.drop("Cabin", axis=1, inplace=True)
train.head()

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="viridis")


sex = pd.get_dummies(train["Sex"], drop_first=True)
sex.head()


embark = pd.get_dummies(train["Embarked"], drop_first=True)
embark.head()

train = pd.concat([train, sex, embark], axis=1)
train.head()

train.drop(["Sex", "Embarked", "Name", "Ticket"], axis=1, inplace=True)
train.head()


train.drop("PassengerId", axis=1, inplace=True)


train.head()


# Now we have a perfect dataframe for a machine learning algorithm
# to do it's thing. One thing to consider is that Pclass is
# actually a categorical column and may have a different
# result if we used get_dummies instead of leaving it for
# the machine learning algorithm to interpret it as a continuous variable.
# I will seperate it after running my algorithm the first time
# in order to find out.

X = train.drop("Survived", axis=1)
y = train["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


logmodel = LogisticRegression(max_iter=999999)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

confusion_matrix(y_test, predictions)


# So what this tells us is that our model is 79% accurate.
# Could be better and so I will be going back and tweak certain
# parameters and information that gets fed into the model
# for an attempt at a higher accuracy. Goal is to be above 90%
