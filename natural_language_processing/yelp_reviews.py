from pyforest import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

yelp = pd.read_csv("yelp.csv")
yelp.head()


yelp.info()


yelp.describe()


yelp["txt_len"] = yelp["text"].apply(len)


g = sns.FacetGrid(data=yelp, col="stars")
g.map(plt.hist, "txt_len", bins=30, ec="black")


sns.boxplot(x="stars", y="txt_len", data=yelp)


sns.countplot(x="stars", data=yelp)


stars = yelp.groupby("stars").mean()


stars


stars.corr()


sns.heatmap(stars.corr(), cmap="coolwarm", annot=True)


# To make this easier to start, I will only take the reviews
# that are either 1 star or 5 star


yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]


X = yelp_class["text"]
y = yelp_class["stars"]


countV = CountVectorizer()


X = countV.fit_transform(X)


X.shape


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)


nb = MultinomialNB()


nb.fit(X_train, y_train)


pred = nb.predict(X_test)


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


pipeline = Pipeline(
    [
        ("bow", CountVectorizer()),
        ("bayes", LogisticRegression(max_iter=999999)),
    ]
)


X = yelp_class["text"]
y = yelp_class["stars"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)


pipeline.fit(X_train, y_train)


pred = pipeline.predict(X_test)


print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# Looks like we were able to make some improvements with logistic regression
