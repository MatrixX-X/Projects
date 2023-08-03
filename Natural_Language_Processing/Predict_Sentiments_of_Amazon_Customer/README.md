# Predicting Sentiments of Amazon Customer Reviews


# About the Project
This project predicts the sentiments of Amazon customer reviews using Natural Language Processing (NLP) and Machine Learning. By analyzing textual data, it classifies reviews into positive and negative sentiments. The process involves data preprocessing, feature engineering, model training, and evaluation. The project aims to provide insights into customer feedback and sentiment trends for Amazon products.

# Requirements
Python, Machine Learning, Natural Language Processing

# Dataset
dataset

# Project Overview
This project aims to predict the sentiments of Amazon customer reviews using Natural Language Processing (NLP) techniques and Machine Learning algorithms. The dataset used for this project is obtained from Amazon product reviews.

<strong>Data Preprocessing:</strong> The initial step involves loading and exploring the dataset, handling missing values, and understanding the data's structure.

<strong>Data Preparation:</strong> We create a new feature called "Helpful%" to gauge the helpfulness of each review. Based on this metric, reviews are categorized into different groups, representing their positivity or negativity.

<strong>Analyzing Upvotes:</strong> The project analyzes the distribution of upvotes across different sentiment scores, providing insights into customer feedback trends.

<strong>Sentiment Prediction:</strong> We apply the Bag of Words technique and TF-IDF to convert textual data into numerical features. A Logistic Regression model is used to predict the sentiments of reviews.

<strong>Top Positive and Negative Words:</strong> We identify the top 20 positive and negative words contributing to sentiments using the trained Logistic Regression model.

<strong>Automation:</strong> Functions are created to automate the preprocessing, model training, and evaluation processes for ease of use.

<strong>Handling Imbalance:</strong> To address data imbalance, we use RandomOverSampler to resample the data.

<strong>Cross Validation:</strong> GridSearchCV is employed to optimize hyperparameters for the Logistic Regression model.

Hope you find this doc helpful :sweat_smile:.

