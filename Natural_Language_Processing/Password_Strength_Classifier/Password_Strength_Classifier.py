#!/usr/bin/env python
# coding: utf-8

# # Password Strength Classifier

# In[1]:


#libraries
import pandas as pd
import numpy as np


# In[2]:


# read data
data = pd.read_csv('data.csv', on_bad_lines='skip')
data


# In[3]:


# extra info
print(data.columns)
print(data.size)
print(data.dtypes)


# In[4]:


# unique data
data['strength'].unique()


# In[5]:


# check total missing data
data.isna().sum()


# In[6]:


# find missing data
data[data['password'].isnull()]


# In[7]:


# drop missing data
data.dropna(inplace = True)


# In[8]:


# again check missing data
data.isnull().sum()


# In[9]:


# best visualization plot
import plotly.express as px
# plot
fig = px.histogram(data, x='strength', color='strength', title='Countplot of Strength')
fig.show()


# **The data has high count of 1, so the data is imbalanced**

# In[10]:


# converting to array data so we can perform on it ,can be imported to model by dataframe also
password_tuple = np.array(data)
password_tuple


# In[11]:


# shuffle data
import random
random.shuffle(password_tuple)


# In[12]:


# split data
X = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]


# In[13]:


#data passed to tfidf must be in the form of character but not in word, as we determine the strength of password based on character
#create a custom function to split input into characters of list
def word_divide(inputs):
    character = []
    for i in inputs:
        character.append(i)
    return character
word_divide('kzde5577')


# In[14]:


# vectorize(give numerical value acc to the character) the characters from string to numerical data 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=word_divide)
X_tf = vectorizer.fit_transform(X)
X_tf.shape


# In[15]:


# 126 features/vector, this method gives all features name
vectorizer.get_feature_names_out()


# In[16]:


# first document (124,)
first_document_vector = X_tf[0]
first_document_vector


# In[17]:


#(change it to (124,1) the .todense transposes the data so add T to neutralize the effect)
first_document_vector.T.todense()


# In[18]:


# build a dataframe in the descending order to use for train test split
data1 = pd.DataFrame(first_document_vector.T.todense(),index = vectorizer.get_feature_names_out(), columns = ['TF-IDF'])
data1.sort_values(by = ['TF-IDF'], ascending = False)


# In[19]:


# split the data in train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tf, y)


# In[20]:


# balancing imbalance data
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler()
X_train_res, y_train_res = os.fit_resample(X_train,y_train)
from collections import Counter
print('Original Dataset shape {}'.format(Counter(y_train)))
print('Resampled Dataset shape {}'.format(Counter(y_train_res)))


# In[21]:


# build logistic model with multiple classes(multinomial)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000)
log.fit(X_train_res, y_train_res)


# In[22]:


# sample test data provied to check if the function works
# convert pass to array then vectorize it
dt = np.array(['%@123abcd'])
pred = vectorizer.transform(dt)
log.predict(pred)


# In[23]:


# predict the value for test data
y_pred = log.predict(X_test)
y_pred


# In[24]:


# check confusion_matrix, accuracy_model, classification_report
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:




