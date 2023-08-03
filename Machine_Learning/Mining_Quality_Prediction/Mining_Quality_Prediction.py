#!/usr/bin/env python
# coding: utf-8

# # TASK #1: UNDERSTAND THE PROBLEM STATEMENT AND BUSINESS CASE
# 

# # Applications in which mining companies leverage the power of Artificial Intelligence and Machine Learning.

# - Mineral Explorations 
# - Autonomous Drillers 
# - Minerals sorting

# # TASK #2: IMPORT LIBRARIES/DATASETS AND PERFORM DATA EXPLORATION 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 


# In[2]:


mining_df = pd.read_csv('mining_data.csv')
mining_df


# In[3]:


mining_df.dtypes


# In[4]:


# check the number of null elements in the dataframe
mining_df.isnull().sum()


# In[5]:


mining_df.describe() # Performs different mathematical operation done on the dataset


# # TASK #3: PERFORM DATA VISUALIZATION

# In[6]:


mining_df.hist(bins = 30, figsize = (20, 20), color = 'r')
plt.show()


# In[7]:


# Obtain the correlation matrix
mining_df.corr()


# In[8]:


plt.figure(figsize = (20, 20))
sns.heatmap(mining_df.corr(), annot = True)
# From this pair plot, we can infer that there is a relationship between iron feed and silica feed 
# Also, a relationship between silica concentrate and iron concentrate.


# - Plotting the scatterplot between % Silica Concentrate and Iron Concentrate and try to relate to the correlation matrix. 
# 
# 
# 

# In[9]:


sns.scatterplot(data = mining_df, x= '% Silica Concentrate', y = '% Iron Concentrate')


# - Plotting the scatterplot between % Iron Feed and % Silica Feed and try to relate to the correlation matrix. 

# In[10]:


sns.scatterplot(data = mining_df, x = '% Iron Feed', y = '% Silica Feed')


# # TASK #4: PREPARE THE DATA BEFORE MODEL TRAINING

# In[11]:


df_iron = mining_df.drop(columns = '% Silica Concentrate')
df_iron_target = mining_df['% Silica Concentrate']


# In[12]:


df_iron.shape


# In[13]:


df_iron_target.shape


# In[14]:


df_iron = np.array(df_iron)
df_iron_target = np.array(df_iron_target)


# In[15]:


# reshaping the array
df_iron_target = df_iron_target.reshape(-1,1)
df_iron_target.shape


# In[16]:


# scaling the data before feeding the model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(df_iron)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(df_iron_target)


# In[17]:


# spliting the data in to test and train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# - The dataset was successful split successfully into train and test set

# In[18]:


X_train.shape


# In[19]:


X_test.shape


# In[20]:


y_train.shape


# In[21]:


y_test.shape


# # TASK #5: TRAIN AND EVALUATE A LINEAR REGRESSION MODEL

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# In[23]:


LinearRegression_model = LinearRegression()
LinearRegression_model.fit(X_train, y_train)


# In[24]:


accuracy_LinearRegression = LinearRegression_model.score(X_test, y_test)
accuracy_LinearRegression


# # TASK #6: TRAIN AND EVALUATE A DECISION TREE AND RANDOM FOREST MODELS

# # Decision Tree Model

# In[25]:


# Decision tree builds regression or classification models in the form of a tree structure. 
# Decision tree breaks down a dataset into smaller subsets while at the same time an associated decision tree is incrementally developed. 
# The final result is a tree with decision nodes and leaf nodes.

from sklearn.tree import DecisionTreeRegressor

DecisionTree_model = DecisionTreeRegressor()
DecisionTree_model.fit(X_train, y_train)


# In[26]:


accuracy_DecisionTree = DecisionTree_model.score(X_test, y_test)
accuracy_DecisionTree


# # Random Forest Model(Ensemble Model)

# In[27]:


# Many decision Trees make up a random forest model which is an ensemble model. 
# Predictions made by each decision tree are averaged to get the prediction of random forest model.
# A random forest regressor fits a number of classifying decision trees on various  sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 


# - Training a Random Forest Regressor Model with n_estimators = 100 and max_depth of 10 

# In[28]:


from sklearn.ensemble import RandomForestRegressor

RandomForest_model = RandomForestRegressor(n_estimators = 100, max_depth = 10)
RandomForest_model.fit(X_train, y_train.ravel())

accuracy_RandomForest = RandomForest_model.score(X_test, y_test)
accuracy_RandomForest


# # TASK #7: TRAIN AN ARTIFICIAL NEURAL NETWORK TO PERFORM REGRESSION TASK

# In[29]:


pip install tensorflow


# In[30]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, amsgrad = False)
ANN_model = keras.Sequential()
ANN_model.add(Dense(250, input_dim = 22, kernel_initializer='normal',activation='relu'))
ANN_model.add(Dense(500,activation = 'relu'))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(1000, activation = 'relu'))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(1000, activation = 'relu'))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(500, activation = 'relu'))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(250, activation = 'relu'))
ANN_model.add(Dropout(0.1))
ANN_model.add(Dense(1, activation = 'linear'))
ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()


# In[31]:


history = ANN_model.fit(X_train, y_train, epochs = 5, validation_split = 0.2)


# In[32]:


result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1 - result
print("Accuracy : {}".format(accuracy_ANN))


# In[33]:


history.history.keys()


# In[34]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


# # TASK #8: COMPARE MODELS AND CALCULATE REGRESSION KPIs 
# 

# In[35]:


# From the above results, it can be seen that, decision tree model out-performs the other models.


# In[36]:


y_predict = DecisionTree_model.predict(X_test)
plt.plot(y_predict, y_test, '^', color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')


# In[37]:


y_test


# In[38]:


y_predict =y_predict.reshape(-1,1)
y_predict


# In[39]:


y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)
plt.plot(y_test_orig, y_predict_orig, "^", color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')


# In[40]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 






