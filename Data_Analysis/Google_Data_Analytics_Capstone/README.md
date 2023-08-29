# Cyclistic Bike Share Data Analysis

This repository contains a Python script for analyzing and processing data from the Cyclistic bike share program. The program reads data from multiple CSV files, combines them, and performs various data cleaning, transformation, and analysis tasks. The resulting insights provide valuable information about ride patterns, member types, and trends over time.

## Requirements
Python, Tableau

## Dataset
[Data](https://divvy-tripdata.s3.amazonaws.com/index.html)


## Project Overview

1. **Data Loading and Combining:** The script loads ride data from multiple CSV files located in a specified directory. It combines these files into a single DataFrame for further analysis.

2. **Data Cleaning:** The script performs data cleaning operations, including removing missing values, dropping unnecessary columns, and eliminating duplicate rows.

3. **Time-based Analysis:** The script extracts useful insights related to time, such as day of the week, hour of the day, and month of the ride. It calculates the duration of each ride in seconds and categorizes rides as day or night based on the start time.

4. **Descriptive Statistics:** The script calculates descriptive statistics such as the mean and maximum ride length, as well as the mode of day of the week.

5. **Visualizations:** The script creates informative visualizations, including box plots to compare ride lengths by member type and scatter plots to visualize ride length in relation to rideable type.

6. **Temporal Trends:** The script analyzes temporal trends by plotting daily and monthly ride counts over time, providing insights into usage patterns.

7. **Pivot Tables:** Pivot tables are used to summarize data across various dimensions, including average ride length for different member types and average ride length by day of the week and member type. Additionally, ride counts are summarized by day of the week and member type.

[Link to Notebook](https://www.kaggle.com/abdulmateenmulla)

[Link to Tableau](https://public.tableau.com/app/profile/abdul.mateen.mulla/viz/GoogleAnalyticsCaseStudyBicycleShareAnalysis/Dashboard1)


## Result

![Dashboard](https://github.com/MatrixX-X/Projects/blob/main/Data_Analysis/Google_Data_Analytics_Capstone/Dashboard.png)
