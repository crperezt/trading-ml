# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import make_scorer
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from datetime import date
from config import FORMS_DATA_PATH
from os.path import join
#from insider_data import SecCrawler

#import final model -done
#Generate dataset for one-year
#Maintain stock symbol
#Classify dataset
#List plus10 and minus 10

def create_dataset(from_first=False):
    #Set date range for last year from today, or this month
    if from_first:
        end_date = date.today().replace(day=1)
        start_date = end_date.replace(year=end_date.year-1)
    else:
        end_date = date.today()
        start_date = end_date.replace(year=end_date.year-1)
    
    end_date = end_date.isoformat()[0:10]
    start_date = start_date.isoformat()[0:10]

    #crawl_new_filings(end_date)

def crawl_new_filings(end_date):
    lf = open(join(FORMS_DATA_PATH,'last-fetched.txt'),'r+')
    last_fetched = lf.readline().strip('\n')
    lf.close()
    print last_fetched

    s = SecCrawler()
    s.parse_form4(last_fetched, end_date)
    lf = open(join(FORMS_DATA_PATH,'last-fetched.txt'),'w')
    lf.write(end_date)

    return

def preprocess_features(X):
	output = pd.DataFrame(index = X.index)
	for col, col_data in X.iteritems():
		if col_data.dtype == object:
			col_data = pd.get_dummies(col_data, prefix = col)
		output = output.join(col_data)

	return output
    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    pd.DataFrame(y_pred).to_csv('y_pred.csv')
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    print "Classification Report: \n" + str(classification_report(target,y_pred))

def pick_stocks():

    dataset = pd.read_csv(join(FORMS_DATA_PATH,"picks_dataset.csv"))
    print "Data read successfully."

    #Cleanup data
    #dataset.dropna(axis=0, inplace=True)
    company_list = pd.Series(dataset['COMPANY'].copy(deep=True))
    dataset = dataset.drop('COMPANY', axis=1)
    dataset['MKTCAP'].replace('.*Small.', 'small', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Mid.', 'mid', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Large.', 'large', inplace=True, regex=True)

    #Extract the feature and target columns
    feature_cols = list(dataset.columns[:-1])
 
    #Show the list of columns
    print "Feature columns:\n{}".format(feature_cols)

    X_all = dataset[feature_cols]
    X_all = preprocess_features(X_all)

    print X_all.head()

    clf = joblib.load('best_model1.pkl')

    y_pred = clf.predict(X_all)

    rd = {'company': company_list, 'category': y_pred} 
    rdf = pd.DataFrame(data=rd)
    print rdf.category[0]
    results = rdf[rdf.category.astype(np.str)==np.str('minus_ten')]
    print results

if __name__ == "__main__":
    pick_stocks()
