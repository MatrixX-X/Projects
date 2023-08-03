# Password Strength Classifier

# About the Project
The Password Strength Classifier project is a machine learning model that aims to predict the strength of passwords based on their character composition. Passwords are an essential aspect of security, and this project helps evaluate the robustness of passwords by classifying them into three categories: weak, medium, and strong.

# Requirements
Python, Machine Learning, Natural Language Processing

# DataSet
data

# Project Overview
The project utilizes a dataset containing passwords and their corresponding strength labels (weak, medium, or strong). The main steps of the project are as follows:

<strong>Data Preprocessing:</strong> The dataset is loaded from a CSV file, and missing data is handled by dropping any rows with missing passwords. The data is then visualized to understand the distribution of password strengths.

<strong>Feature Engineering:</strong> To prepare the data for modeling, the passwords are converted into numerical representations using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorization technique. This process transforms each password into a vector of numerical values based on the frequency of individual characters.

<strong>Data Balancing:</strong> As the original dataset is imbalanced, where weak passwords are dominant, a Random Over-Sampling technique is applied to balance the data by creating synthetic samples of the minority classes.

<strong>Model Building:</strong> The Logistic Regression algorithm is chosen as the classification model due to its ability to handle multiple classes (multinomial). The model is trained using the balanced dataset.

<strong>Model Evaluation:</strong> The model's performance is evaluated using a test dataset, and metrics such as confusion matrix, accuracy, and classification report are used to assess its effectiveness in predicting password strengths.

Hope you find this doc helpful :sweat_smile:.


