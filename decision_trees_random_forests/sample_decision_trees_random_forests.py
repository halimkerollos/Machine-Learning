# Using a small dataset on Kyphosis,
# I will create a decision trees and random foresets to predict
# whether or not the corrective spine surgery was successful


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
from IPython.display import Image
from io import StringIO
import pydot

# get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv("kyphosis.csv")
df.head()


# Kyphosis is a spinal condition that requires surgery.
# Column descriptions: under Kyphosis, we have values that tell
# us if the condition was present or absent after the operation.
# Age in months is the nect columns.
# Number is the number of vertebra involved in the operation.
# Start is the number of the first or top-most vertebra operated on
df.info()
df.describe()


sns.pairplot(df, hue="Kyphosis")


X = df.drop("Kyphosis", axis=1)
y = df["Kyphosis"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


dtree = DecisionTreeClassifier()


dtree.fit(X_train, y_train)


pred = dtree.predict(X_test)


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


rfc = RandomForestClassifier(n_estimators=200)


rfc.fit(X_train, y_train)


rfc_pred = rfc.predict(X_test)


print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))


# A bit better but still pretty low accuracy, precision, and recall.
# A reason for this could be that it's very small dataset
# and the train test split might have been unideal.
# The second potential issue is the heavy imbalance of labeled data,
# as I will demonstrate better in the next cell.
# But basically there are far more data points lebeled 'absent'
# than 'present' which combined with the small dataset and random
# state might have given us low valued metrics


df["Kyphosis"].value_counts()


absent = df["Kyphosis"].value_counts()[0]
absent / sum(df["Kyphosis"].value_counts())


# So 79% of our data is labeled 'absent' which is a very poorly
# balanced dataset.
# But at the very least it looks like the random forests
# certainly performed better than a single decision tree and I
# am looking forward to applying this to larger,
# more complex datasets.

# Below I will show two different ways to plot our decision trees,
# so we have a little more visualization on the mathmatics
features = list(df.columns[1:])
features


dot_data = StringIO()
export_graphviz(
    dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True
)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())


plt.figure(figsize=(30, 20))
tree.plot_tree(dtree)
