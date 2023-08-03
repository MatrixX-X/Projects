#!/usr/bin/env python
# coding: utf-8

# # Predict Sentiments of Amazon Customer

# ## Q.Preprocessing

# In[1]:


# libraries
import pandas as pd          # for data manipulation and data analysis
import numpy as np           # for large and multi dimensional array


# In[2]:


# load data
data = pd.read_csv('Reviews.csv')
data


# In[3]:


print(data.columns)
print(data.size)
print(data.dtypes)
print(data.shape)


# ## Data Preparation

# In[4]:


data['Helpful%'] = np.where(data['HelpfulnessDenominator']>0, data['HelpfulnessNumerator']/data['HelpfulnessDenominator'], -1)
data['Helpful%']


# **add different label according to the values**

# In[5]:


data['Helpful%'].unique()


# In[6]:


data['%Upvote'] = pd.cut(data['Helpful%'], bins = [-1,0,0.2,0.4,0.6,0.8,1], labels=['Empty','0-20%', '20-40%', '40-60%', '60-80%', '80-100%'])
data['%Upvote']


# In[7]:


data


# ## Q.Analyze upvotes for diff scores

# In[8]:


data.groupby(['Score','%Upvote']).agg('count')


# In[9]:


data_s = data.groupby(['Score','%Upvote']).agg({'Id':'count'}).reset_index()
data_s


# ## Q.Create pivot table and heatmap

# In[10]:


pivot = data_s.pivot(index = '%Upvote',columns='Score')
pivot


# In[11]:


import seaborn as sns


# In[12]:


sns.heatmap(pivot, annot = True, cmap='YlGnBu');


# ## Q.Apply Bag of Words on data

# In[13]:


data['Score'].unique()


# In[14]:


df1 = data[data['Score'] != 3]
df1


# In[15]:


# score is the dependent variable here
X = df1['Text']


# In[16]:


y_dict={1:0,2:0,4:1,5:1}
y = df1['Score'].map(y_dict)


# In[17]:


#convert text to vector using NLP
from sklearn.feature_extraction.text import CountVectorizer 


# In[18]:


#after countvectorizeration the feature no. changes from x to 114969
c = CountVectorizer(stop_words='english')
X_c = c.fit_transform(X)
X_c.shape


# ## Q.Check Model Accuracy

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


# default training size = 0.75
X_train, X_test, y_train, y_test = train_test_split(X_c, y)


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


log = LogisticRegression(max_iter=1000)
ml = log.fit(X_train,y_train)


# In[23]:


ml.score(X_test,y_test)


# ## Q.Fetch Top 20 positive words & Top 20 negative words

# In[24]:


w = c.get_feature_names_out()
w


# In[25]:


coef = ml.coef_.tolist()[0]
coef


# In[26]:


coef_data = pd.DataFrame({'Word':w, 'Coefficient':coef})
coef_data


# In[27]:


coef_data = coef_data.sort_values(['Coefficient', 'Word'], ascending=False)
coef_data


# In[28]:


coef_data.head(20)


# In[29]:


coef_data.tail(20)


# ## Q.Automate the previous 3 tasks

# In[30]:


def text_fit(X,y,nlp_model,ml_model, coef_show=1):
    X_c = nlp_model.fit_transform(X)
    print('features:{}'.format(X_c.shape[1]))
    
    X_train, X_test, y_train, y_test = train_test_split(X_c, y)
    ml=ml_model.fit(X_train,y_train)
    acc = ml.score(X_test,y_test)
    print(acc)
    
    if coef_show == 1:
        w = c.get_feature_names_out()
        coef = ml.coef_.tolist()[0]
        coef_data = pd.DataFrame({'Word':w, 'Coefficient':coef})
        coef_data = coef_data.sort_values(['Coefficient', 'Word'], ascending=False)
        print('\n')
        print('--Top 20 Positive Words--')
        print(coef_data.head(20))
        print('\n')
        print('--Top 20 Negative Words--')
        print(coef_data.tail(20))


# In[31]:


text_fit(X,y,c,log)


# ## Q.Automate the Predictions

# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[33]:


def predict(X,y,nlp_model,ml_model):
    X_c = nlp_model.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_c, y)
    ml=ml_model.fit(X_train,y_train)
    predictions = ml.predict(X_test)
    cm = confusion_matrix(predictions, y_test)
    print(cm)
    acc = accuracy_score(predictions, y_test)
    print(acc)


# In[34]:


c = CountVectorizer()


# In[35]:


predict(X,y,c,log)


# ## Q.Apply more NLP & ML on data

# In[36]:


from sklearn.dummy import DummyClassifier


# In[37]:


#count vetorization(no parameter) and dummy Classifier
text_fit(X,y,c,DummyClassifier(),0)


# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[39]:


tfidf = TfidfVectorizer(stop_words='english')
text_fit(X,y,tfidf,log,0)


# In[40]:


#tf-idf and Logistic regression
predict(X,y,tfidf,log)


# ## Q.Data Preparation for predicting the Upvotes

# In[41]:


df2 = data[data['Score']==5]
df2


# In[42]:


data2 = df2[df2['%Upvote'].isin(['80-100%', '60-80%', '40-60%', '20-40%'])]
data2


# In[43]:


X = data2['Text']
X


# In[44]:


y_dict = {'80-100%':1, '60-80%':1, '20-40%':0, '40-60%':0}
y = data2['%Upvote'].map(y_dict)
y


# In[45]:


y.value_counts()


# ## Q.Apply Tf-Idf on data

# In[46]:


tfidf = TfidfVectorizer(stop_words='english')
X_c = tfidf.fit_transform(X)
X_c.shape


# In[47]:


y.value_counts()


# ## Q.Handle Imbalace data if data is Imbalance

# In[48]:


# requires Tensorflow
#pip install tensorflow


# In[49]:


from imblearn.over_sampling import RandomOverSampler


# In[50]:


os = RandomOverSampler()


# In[51]:


X_train_res, y_train_res = os.fit_resample(X_c,y)


# In[52]:


from collections import Counter


# In[53]:


print('Original Dataset shape {}'.format(Counter(y)))
print('Resampled Dataset shape {}'.format(Counter(y_train_res)))


# ## Q.Do Cross validation using GridSearchCV & then do predictions

# In[54]:


from sklearn.model_selection import GridSearchCV


# In[55]:


q = np.arange(-2,3)
q


# In[56]:


grid = {'C' : 10.0 **q,'penalty':['l2']}


# In[57]:


# n_jobs = -1 use all CPU resources
clf = GridSearchCV(estimator=log, param_grid=grid, cv = 5, n_jobs=-1, scoring='f1_macro')


# In[58]:


clf.fit(X_train_res, y_train_res)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X_c,y) # since the number of features have changed, split again


# In[60]:


pred = clf.predict(X_test)
pred


# ## Q.Checking Accuracy of cross validated model

# In[61]:


from sklearn.metrics import confusion_matrix


# In[62]:


confusion_matrix(y_test, pred)


# In[63]:


accuracy_score(y_test, pred)

