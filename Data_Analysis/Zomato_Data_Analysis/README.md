# Zomato Data Analysis 

# About the Project
This project provides a comprehensive analysis of Zomato restaurant data using Python's pandas, numpy, matplotlib, seaborn, Plotly, and Folium libraries. The project aims to understand various aspects of the Zomato dataset, from data cleaning and preprocessing to in-depth insights and visualization. 

# Requirements
Python

# Dataset
zomato.zip

# Project Overview
The project can be divided into several sections:

### Project Setup and Data Reading
- The project starts with importing necessary libraries and reading the Zomato restaurant data from a CSV file using pandas.

### Data Cleaning
- Identifying missing values and analyzing their distribution across features.
- Presenting the percentage of missing values in each feature.

### Data Preprocessing
- Converting the 'rate' column to a numerical format and handling values like 'NEW' and '-'.
- Converting the 'approx_cost(for two people)' column to an integer format.
  
## Analyzing and Visualizing Data
### Restaurant Ratings and Distribution
- Calculating and visualizing the average rating of each restaurant.
- Creating a histogram to visualize the distribution of average ratings.

![1](https://github.com/MatrixX-X/Projects/blob/main/Data_Analysis/Zomato_Data_Analysis/1.png)

### Most Popular Restaurants
- Visualizing the top restaurant chains in Bangalore based on the number of outlets.

![1](https://github.com/MatrixX-X/Projects/blob/main/Data_Analysis/Zomato_Data_Analysis/2.png)
- Creating pie charts to display the proportion of restaurants accepting online orders and those that don't.

### Booking Tables
- Creating a pie chart to display the ratio of restaurants that provide table booking and those that don't.

### Types of Restaurants
- Analyzing the types of restaurants using bar plots, revealing the dominance of "Quick Bites" type restaurants.

### Highest Voted Restaurants
- Displaying the bar chart of the highest-voted restaurants.

### Restaurants Across Different Locations
- Displaying the top 10 locations with the most number of restaurants using various visualization methods.

### Cuisines Analysis
- Analyzing the top cuisines and displaying them using a bar plot.

### Approximate Cost Analysis
- Analyzing and visualizing the approximate cost for two people using histograms and scatter plots.
- Comparing the votes and costs for restaurants accepting online orders and those that don't.

### Luxury and Budget Restaurants
- Displaying the most expensive and cheapest restaurants using bar plots.

### Affordable and Highly Rated Restaurants
- Identifying affordable and highly rated restaurants.
- Analyzing and visualizing such restaurants based on ratings and costs.

### Geographic Analysis
- Obtaining latitude and longitude coordinates for each restaurant location in Bangalore.
- Creating a heatmap to visualize the concentration of restaurants across the city.

### Geographic Analysis of North Indian Restaurants
- Creating a heatmap specifically for North Indian restaurants' locations.

### Analysis of Restaurant Reviews
- Analyzing and visualizing reviews from restaurant reviews_list using word clouds.

### Functionality for Further Analysis
- Creating functions to generate word clouds for restaurant reviews and dish likes based on the restaurant type.

