#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


get_ipython().run_line_magic("matplotlib", "inline")
cancer = load_breast_cancer()

cancer.keys()
print(cancer["DESCR"])
print(cancer["feature_names"])

df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])
df.head()

col = ["target"]
target = pd.DataFrame(cancer["target"], columns=col)
target.info()


cdf = df.join(target)

X = cdf.drop("target", axis=1)
y = cdf["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


scaler = StandardScaler()

scaler.fit(X_train)
strain = scaler.transform(X_train)
strain

stest = scaler.transform(X_test)
stest

strain.shape
stest.shape

pca = PCA(n_components=2)
pca.fit(strain)

strain_pca = pca.transform(strain)
stest_pca = pca.transform(stest
strain_pca.shape
stest_pca.shape


plt.figure(figsize=(10, 4))
plt.scatter(strain_pca[:, 1], strain_pca[:, 0], c=y_train)
plt.xlabel("First Principle Component")
plt.ylabel("Second Principal Component")

pca.components_

logR = LogisticRegression()

logR.fit(strain_pca, y_train)

pred = logR.predict(stest_pca)


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))