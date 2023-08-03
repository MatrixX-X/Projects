#!/usr/bin/env python
# coding: utf-8

# ## Scenario : Based on the headline will the stock price increase(1) or decrease/constant(0)
# ## Stock Sentiment Analysis using News Headlines

# In[1]:


# import libraries
import pandas as pd
import numpy as np


# In[2]:


# read data
data = pd.read_csv('Data.csv', encoding='ISO-8859-1')
data


# In[3]:


# extra info
print(data.columns)
print(data.dtypes)


# In[4]:


# total mising data
data.isna().sum()


# In[5]:


# drop missing data
data.dropna(inplace=True)
data.reset_index(drop = True, inplace = True)


# In[6]:


# check again
data.isna().sum()


# In[7]:


# spliting data in train and test
train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']


# In[8]:


# dropping date and label since it is not useful
data = train.drop(columns = ['Date','Label'], axis = 0)
data.head()


# In[9]:


data.replace('[^a-zA-Z]',' ', inplace = True)
data


# In[10]:


new_index = [str(i) for i in range(25)]
new_index


# In[11]:


data.columns = new_index
data.head()


# In[12]:


data.index


# In[13]:


for index in new_index:
    data[index] = data[index].str.lower()
data.head(1)


# In[14]:


headlines = []
for i in data.iloc[1,0:25]:
    headlines.append(i)
' '.join(headlines)


# In[15]:


# list comprehension same output but diff command with compariosion to upper code
' '.join([str(i) for i in data.iloc[1,0:25]])


# In[16]:


headlines = []
for row in range(0,len(data)):
    headlines.append(' '.join([str(i) for i in data.iloc[row,0:25]]))
    


# In[17]:


headlines[0:3]


# In[18]:


# implement bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[19]:


cv = CountVectorizer(ngram_range=(2,2))
traindata_x = cv.fit_transform(headlines)


# In[20]:


# build model and implement
classifier = RandomForestClassifier(n_estimators= 200, criterion= 'entropy')
classifier.fit(traindata_x, train['Label'])


# In[21]:


# bring the test data together
test_transform = []
for row in range(0, len(test)):
    test_transform.append(' '.join(str(i) for i in test.iloc[row,2:27]))


# In[22]:


# transform the data
test_data = cv.transform(test_transform)


# In[23]:


predictions = classifier.predict(test_data)


# In[24]:


from sklearn.metrics import confusion_matrix


# In[25]:


cm = confusion_matrix(test['Label'], predictions)
cm


# In[26]:


# plot
from sklearn import metrics
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=["positive", "negative"]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    


# In[27]:


plt.figure()
cm=confusion_matrix(test['Label'],predictions)
print(cm)
plot_confusion_matrix(cm)    
plt.show()


# In[28]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[33]:


acc = accuracy_score(test['Label'], predictions)
print(acc)
report = classification_report(test['Label'], predictions)
print(report)


# In[34]:


from sklearn.naive_bayes import MultinomialNB


# In[36]:


nb = MultinomialNB()
nb.fit(traindata_x, train['Label'])

predictions = nb.predict(test_data)
matrix = confusion_matrix(test['Label'], predictions)
print(matrix)
acc = accuracy_score(test['Label'], predictions)
print(acc)
report = classification_report(test['Label'], predictions)
print(report)


# In[ ]:




