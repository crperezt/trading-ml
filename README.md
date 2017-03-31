# Predicting Stock Prices Based on Insider Trading SEC Data

This is a final project for the Udacity Machine Learning Engineer Nanodegree. The scripts in this repository can be used to crawl Form 4 data from the SEC EDGAR database, create a particular dataset based on the SEC data and stock price return data, and train a machine learning model to predict stock price movements based on the dataset.

Requires Numpy, Pandas and Scikit-Learn to be installed.

# Files:

Project Notebook.ipynb - iPython Notebook describing the overall project.

form4_pull.py: retrieves all Form 4 forms for companies listed in company_list.csv from Jan 2010 through Dec 2016 (over 7GB of data).

form4_pull_mt.py: rudimentary multi-threaded version of form4_pull.py.

sec_crawler.py: helper functions for form4_pull.py, adapted from https://github.com/rahulrrixe/SEC-Edgar

sec_crawler_mt.py: helper functions for form4_pull_mt.py

create_dataset.py: creates a dataset (as described in FinalReport.docx) from the Form 4 data. Also pulls stock price returns for each company for inclusion in the dataset. Outputs dataset to dataset.csv.

trading.py: trains a Naive Bayes classifier using dataset.csv. Outputs the model to model.pkl.

stock_picker.py: predicts the stock price return category for each company from Jan 2017-Dec 2017, based on data from picker_dataset.csv, which includes data from Jan2016-Dec2016. Categories are ABOVE (10% higher than S&P500), MIDDLE (around -10% to 10% of S&P500), BELOW (less than -10% of S&P500). Outputs results to picks.txt, sorted by prediction confidence (descending order).