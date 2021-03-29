#!/usr/bin/env python
# coding: utf-8

# In[65]:


from pyforest import *


# In[66]:


#nltk.download_shell() used to download stopwords


# In[67]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[68]:


print(len(messages))


# In[69]:


messages[0]


# In[70]:


for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')


# In[71]:


messages[0]


# the \t indicates it is a tab separated value(TSV), instead of a comma sep value(CSV)

# In[72]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', 
                       sep='\t', names=['label', 'message'])


# In[73]:


messages


# In[74]:


messages.describe()


# In[75]:


messages.groupby('label').describe()


# In[76]:


messages['length'] = messages['message'].apply(len)


# In[77]:


messages


# In[78]:


messages['length'].plot.hist(bins=200, ec='black')


# In[79]:


messages['length'].describe()


# In[80]:


messages[messages['length'] == 910]['message'].iloc[0]


# In[81]:


messages.hist(column='length', by='label', bins=150, figsize=(12,4), ec='black')


# Looks from the plot above that length of message could be a great indicator of whether or not a message is ham or spam as spam messages tend to be longer than ham.
# Next we'll convert the texts into bags of words vectors (sequence of numbers) to determine similarities and further help in our spam detection software build.

# In[82]:


mess = 'Sample message! Notice: it has punctuation.'


# In[83]:


import string


# In[84]:


string.punctuation


# In[85]:


nopunc = [c for c in mess if c not in string.punctuation]


# In[86]:


nopunc


# In[87]:


from nltk.corpus import stopwords


# In[88]:


stopwords.words('english')


# In[89]:


nopunc = ''.join(nopunc)
nopunc


# In[90]:


nopunc.split()


# In[91]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[92]:


clean_mess


# In[93]:


def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[94]:


messages['message'].head(5).apply(text_process)


# In[107]:


messages['message'].head(5)


# In[108]:


from sklearn.feature_extraction.text import CountVectorizer


# In[109]:


bow_transformer = CountVectorizer(analyzer=text_process, ).fit(messages['message'])


# In[110]:


print(len(bow_transformer.vocabulary_))


# In[122]:


mess5 = messages['message'][4]


# In[124]:


print(mess5)


# In[127]:


bow5 = bow_transformer.transform([mess5])


# In[128]:


print(bow5)
print(bow5.shape)


# In[133]:


print(bow_transformer.get_feature_names()[2948])
print(bow_transformer.get_feature_names()[4777])
print(bow_transformer.get_feature_names()[6123])
print(bow_transformer.get_feature_names()[6877])
print(bow_transformer.get_feature_names()[7842])
print(bow_transformer.get_feature_names()[10433])
print(bow_transformer.get_feature_names()[10450])
print(bow_transformer.get_feature_names()[10799])


# In[136]:


messages_bow = bow_transformer.transform(messages['message'])


# In[137]:


print("Shape of sparse matrix: ", messages_bow.shape)


# In[138]:


messages_bow.nnz


# In[139]:


sparsity = (100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print("Sparsity: {}".format(sparsity))


# In[140]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[141]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[148]:


tfidf_transformer.idf_[bow_transformer.vocabulary_['cat']]


# In[149]:


messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[150]:


from sklearn.naive_bayes import MultinomialNB


# In[151]:


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[154]:


spam_detect_model.predict(tfidf4)[0]


# In[155]:


messages['label'][3]


# In[156]:


all_pred = spam_detect_model.predict(messages_tfidf)


# In[157]:


all_pred


# In[158]:


from sklearn.model_selection import train_test_split


# In[159]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)


# In[160]:


from sklearn.pipeline import Pipeline


# In[161]:


from sklearn.ensemble import RandomForestClassifier


# In[172]:


pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', RandomForestClassifier())
                    ])


# In[173]:


pipeline.fit(msg_train, label_train)


# In[174]:


pred = pipeline.predict(msg_test)


# In[175]:


from sklearn.metrics import classification_report


# In[176]:


print(classification_report(label_test, pred))


# Looks to me the random forest classifier performed better than the multinomial naive bayes.
