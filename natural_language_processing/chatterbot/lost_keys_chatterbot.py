#!/usr/bin/env python
# coding: utf-8

# In[23]:


from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer


# In[24]:


bot = ChatBot('Lex')


# In[25]:


bot = ChatBot('Lex',
             storage_adapter='chatterbot.storage.SQLStorageAdapter',
             database_url='sqlite:///database.sqlite3'
             )


# In[26]:


bot = ChatBot('Lex',
             logic_adapters=[
                 'chatterbot.logic.BestMatch',
                 'chatterbot.logic.TimeLogicAdapter'
             ])


# In[27]:


trainer = ListTrainer(bot)


# In[28]:


trainer.train([
    'Hi',
    'Hello',
    'Did you check your pockets?',
    'I lost my keys too',
    'Where is the last place you remember seeing them?',
    "I can't find my keys either...",
    "That's unfortunate, where do you think they could've gone?",
    'What exactly can I do about that?',
    "Try looking between the couch cusions.",
    "I guess they're gone then",
    "Thanks! You really made my day!"
])


# In[33]:


response = bot.get_response('I lost my keys')


# In[34]:


print("Bot response: ", response)


# In[ ]:


name=input('Enter your name')
print("Welcome to the Unhelpful Bot Service {}!, Let me know what I can do for you...so I can ignore it.".format(name))
while True:
    request=input(name+':')
    if request=='Bye' or request=='bye':
        print('Bot: Bye')
        break
    else:
        response=bot.get_response(request)
        print('Bot: ', response)


# In[ ]:




