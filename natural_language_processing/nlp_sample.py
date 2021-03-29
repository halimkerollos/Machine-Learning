from pyforest import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import string

# nltk.download_shell() used to download stopwords


messages = [
    line.rstrip() for line in open("smsspamcollection/SMSSpamCollection")
]


print(len(messages))


messages[0]


for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print("\n")


messages[0]


# the \t indicates it is a tab separated value(TSV),
# instead of a comma sep value(CSV)


messages = pd.read_csv(
    "smsspamcollection/SMSSpamCollection",
    sep="\t",
    names=["label", "message"],
)


messages


messages.describe()


messages.groupby("label").describe()


messages["length"] = messages["message"].apply(len)


messages


messages["length"].plot.hist(bins=200, ec="black")


messages["length"].describe()


messages[messages["length"] == 910]["message"].iloc[0]


messages.hist(
    column="length", by="label", bins=150, figsize=(12, 4), ec="black"
)


# Looks from the plot above that length of message
# could be a great indicator of whether or not a message is
# ham or spam as spam messages tend to be longer than ham.
# Next we'll convert the texts into bags of words vectors
# (sequence of numbers) to determine similarities and further
# help in our spam detection software build.


mess = "Sample message! Notice: it has punctuation."


string.punctuation


nopunc = [c for c in mess if c not in string.punctuation]


nopunc


stopwords.words("english")


nopunc = "".join(nopunc)
nopunc


nopunc.split()


clean_mess = [
    word
    for word in nopunc.split()
    if word.lower() not in stopwords.words("english")
]


clean_mess


def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    return [
        word
        for word in nopunc.split()
        if word.lower() not in stopwords.words("english")
    ]


messages["message"].head(5).apply(text_process)


messages["message"].head(5)


bow_transformer = CountVectorizer(
    analyzer=text_process,
).fit(messages["message"])


print(len(bow_transformer.vocabulary_))


mess5 = messages["message"][4]


print(mess5)


bow5 = bow_transformer.transform([mess5])


print(bow5)
print(bow5.shape)


print(bow_transformer.get_feature_names()[2948])
print(bow_transformer.get_feature_names()[4777])
print(bow_transformer.get_feature_names()[6123])
print(bow_transformer.get_feature_names()[6877])
print(bow_transformer.get_feature_names()[7842])
print(bow_transformer.get_feature_names()[10433])
print(bow_transformer.get_feature_names()[10450])
print(bow_transformer.get_feature_names()[10799])


messages_bow = bow_transformer.transform(messages["message"])


print("Shape of sparse matrix: ", messages_bow.shape)


messages_bow.nnz


sparsity = (
    100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])
)
print("Sparsity: {}".format(sparsity))


tfidf_transformer = TfidfTransformer().fit(messages_bow)


tfidf_transformer.idf_[bow_transformer.vocabulary_["cat"]]


messages_tfidf = tfidf_transformer.transform(messages_bow)


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages["label"])


spam_detect_model.predict(tfidf4)[0]


messages["label"][3]


all_pred = spam_detect_model.predict(messages_tfidf)


all_pred


msg_train, msg_test, label_train, label_test = train_test_split(
    messages["message"], messages["label"], test_size=0.3
)


pipeline = Pipeline(
    [
        ("bow", CountVectorizer(analyzer=text_process)),
        ("tfidf", TfidfTransformer()),
        ("classifier", RandomForestClassifier()),
    ]
)


pipeline.fit(msg_train, label_train)


pred = pipeline.predict(msg_test)


print(classification_report(label_test, pred))


# Looks to me the random forest classifier performed
# better than the multinomial naive bayes.
