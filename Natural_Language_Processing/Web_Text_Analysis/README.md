# Web Text Analysis

# About the Project
This Python script is designed to extract and analyze textual content from a list of URLs stored in an Excel file. The extracted content is then subjected to various linguistic analyses to derive insights about sentiment, readability, and linguistic complexity. The analysis results are stored in an output Excel file for further examination.

# Requirements
Python, Machine Learning, Natural Language Processing

# DataSet
Excel file with links to web pages

# Project Overview
### Features
- <strong>Web Page Content Extraction:</strong> The script utilizes the `requests` library to fetch the HTML content of specified URLs and `BeautifulSoup` for parsing the content, extracting article titles, and article text.
- <strong>Sentiment Analysis:</strong> The script performs sentiment analysis using a pre-defined list of positive and negative words. It calculates positive and negative scores, polarity score, and subjectivity score for each text.
- <strong>Readability Analysis:</strong> The script calculates various readability metrics, including average sentence length, percentage of complex words, and Gunning Fog Index to assess the text's readability.
- <strong>Complexity Analysis:</strong> The script counts complex words in the text based on the number of syllables and identifies personal pronouns.
- <strong>Data Processing:</strong> The script employs the Natural Language Toolkit (NLTK) for tokenization, stop word removal, and other text processing tasks.
- <strong>Output Generation:</strong> Analysis results are compiled and saved in an output Excel file, presenting a comprehensive overview of each text's linguistic characteristics.

### How to Use

1. Prepare an Excel file named `Input.xlsx` containing URLs to be analyzed.
2. Ensure the `MasterDictionary` folder contains positive and negative word lists.
3. Place stopwords files in the `StopWords` directory.
4. Run the script.
5. Analysis results will be saved in `Output.xlsx`.

