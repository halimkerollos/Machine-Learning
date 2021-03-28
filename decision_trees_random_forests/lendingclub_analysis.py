# Lending Club is a service which connects people who need a loan
# with investors.
# I will be looking at publicly available data from LendingClub.com
# and applying decision trees to predict loan defaults.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("loan_data.csv")
df.head()

df.info()
df.describe()


# Checking on the fico scores of individuals with respect
# to the credit policy they were given.
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
df[df["credit.policy"] == 1]["fico"].hist(
    bins=30, label="Credit Policy: 1", alpha=0.8
)
df[df["credit.policy"] == 0]["fico"].hist(
    bins=30, label="Credit Policy: 0", alpha=0.8
)
plt.legend()


plt.figure(figsize=(12, 6))
sns.histplot(
    data=df[df["not.fully.paid"] == 0]["fico"],
    bins=30,
    label="Not Fully Paid: 0",
    alpha=0.8,
    kde=True,
)
sns.histplot(
    data=df[df["not.fully.paid"] == 1]["fico"],
    bins=30,
    label="Not Fully Paid: 1",
    alpha=0.7,
    color="red",
    kde=True,
)
plt.legend()


plt.figure(figsize=(12, 6))
sns.histplot(
    data=df[df["not.fully.paid"] == 1]["fico"],
    bins=30,
    label="Not Fully Paid: 1",
    alpha=0.7,
    color="red",
    kde=True,
)
plt.legend()


# People who have not fully paid seem to number greatest right
# under a 700 fico score.
# Lets explore the loan purposes with a hue on fully  paid.


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="purpose", hue="not.fully.paid")


plt.figure(figsize=(12, 6))
sns.jointplot(
    data=df,
    x="fico",
    y="int.rate",
    kind="kde",
    fill=True,
)


sns.lmplot(
    data=df, x="fico", y="int.rate", hue="credit.policy", col="not.fully.paid"
)


df.info()


# I will have to prepare dummy variables for the purpose
# column as it is categorical


cat_feats = ["purpose"]


final = pd.get_dummies(df, columns=cat_feats, drop_first=True)


final.head()


# Now it's time to split our data and train the model:


X = final.drop("not.fully.paid", axis=1)
y = final["not.fully.paid"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# Creating a prediction and checking it's effectiveness
# with sklearn metrics:


pred = dtree.predict(X_test)


print(confusion_matrix(y_test, pred))
print("\n")
print(classification_report(y_test, pred))


# Looks like we have an accuracy rate of 73% not terrible
# but lets see how it compares with random forests.


rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)


rfc_pred = rfc.predict(X_test)


print(confusion_matrix(y_test, rfc_pred))
print("\n")
print(classification_report(y_test, rfc_pred))


# Definite increase in metrics and quality of model fit,
# with 84% accuracy now, however the recall value for
# 'not.fully.paid==1' did much worse in the random forsts model,
# being 1% instead of 20% for the dtree.
# Which model we would use really depends on the specific questions
# we are trying to answer and that would normally be guided by
# some business goal in mind. Though overall accuracy was better
# with random forests.
