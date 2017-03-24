# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, classification_report, r2_score
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV 
from sklearn.metrics import make_scorer
from sklearn.cross_validation import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import brier_score_loss

def preprocess_features(X):
	output = pd.DataFrame(index = X.index)
	for col, col_data in X.iteritems():
		if col_data.dtype == object:
			col_data = pd.get_dummies(col_data, prefix = col)
		output = output.join(col_data)

	return output

def performance_metric(y_true, y_pred):
	return f1_score(y_true,y_pred,average='weighted')

def reg_performance_metric(y_true, y_pred):
    return r2_score(y_true,y_pred)

def train_reg(reg, X_train, y_train, X_test, y_test):
    reg.fit(X_train, y_train) 
    scoring_fnc = make_scorer(performance_metric)
    predict_ret(reg,X_test, y_test)

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train) 
    scoring_fnc = make_scorer(performance_metric)
    predict_labels(clf,X_test, y_test)

    prob_pos_clf = clf.predict_proba(X_test)[:,0]


    sorted_prob = zip(clf.predict(X_test),y_test,prob_pos_clf)
    sorted_prob_df = pd.DataFrame(sorted_prob)
    print sorted_prob_df.head()

    sorted_prob_df.sort(2,ascending=False,inplace=True)
    print sorted_prob_df.head(50)
    print("Brier scores: (the smaller the better)")
    clf_score = brier_score_loss(y_test, prob_pos_clf, pos_label='above')
    print("No calibration: %1.3f" % clf_score)

def train_classifierGS(clf, X_train, y_train, params=None):
    cv_iters = 2
    cv_sets = ShuffleSplit(X_train.shape[0], n_iter=cv_iters, test_size=0.20, random_state=0)
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(clf, params, scoring=scoring_fnc, cv=cv_sets)
    grid.fit(X_train, y_train)
    print "best_params_ for the optimal model are: {}.".format(str(grid.best_params_))
    return grid.best_params_, grid.best_estimator_

def train_classifierCV(clf, X_train, y_train, fit_params=None):
    ''' Fits a classifier to the training data. '''

    cv_iters = 10
    cv_sets = ShuffleSplit(X_train.shape[0], n_iter=cv_iters, test_size=0.20, random_state=0)
    scoring_fnc = make_scorer(performance_metric)  

    start = time()
    scores_array = cross_val_score(clf, X_train, y=y_train, fit_params=fit_params, scoring=scoring_fnc, cv=cv_sets)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)
    print "\nCross-Validation scores for " + str(cv_iters) + " iterations:\n"
    print scores_array
    
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

def predict_ret(reg, features, target):
    ''' Makes predictions using a fit classifier. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = reg.predict(features)
    #pd.DataFrame(y_pred).to_csv('y_pred.csv')
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    print "R2 Score: \n" + str(r2_score(target,y_pred))

def count_zeroes(x):
    count = 0
    for i in x[0:12]:
        if i == 0.0:
            count = count + 1
    return count


def run_models(regression=False):

    dataset = pd.read_csv("picker_dataset.csv")
    print "Data read successfully."

    #Cleanup data

    dataset = dataset.drop(['RET1'], axis=1)
    dataset = dataset.drop(['RET6'], axis=1)
    dataset = dataset.drop(['RET12'], axis=1)
    dataset = dataset[dataset.MONTH == '2016-01']
    dataset = dataset.drop(['MONTH'], axis=1)
    dataset['MKTCAP'].replace('.*Small.', 'small', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Mid.', 'mid', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Large.', 'large', inplace=True, regex=True)
    X_company = dataset['COMPANY']
    dataset = dataset.drop(['COMPANY'], axis=1)

    #eliminates points with many starting zeroes
    #dataset = dataset[dataset.apply(lambda x: count_zeroes(x) < 4,axis=1)]

    #Extract the feature columns
    feature_cols = list(dataset.columns)


    #Show the list of columns
    print "Feature columns:\n{}".format(feature_cols)

    #Separate data into feature data and target data
    X_all = dataset[feature_cols]

    X_all = preprocess_features(X_all)
    print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

    clf = joblib.load('bagging_nb_bear_final.pkl')

    y_pred = clf.predict(X_all)

    #rd = {'company': company_list, 'category': y_pred} 
    #rdf = pd.DataFrame(data=rd)
    #print rdf.category[0]
    #results = rdf[rdf.category.astype(np.str)==np.str('minus_ten')]
    #print results


    prob_above_clf = clf.predict_proba(X_all)[:,0]

    sorted_prob = zip(X_company,clf.predict(X_all),prob_above_clf)
    sorted_prob_df = pd.DataFrame(sorted_prob)
    print sorted_prob_df.head()

    sorted_prob_df.sort(2,ascending=False,inplace=True)
    sorted_prob_df.to_csv('bear_picks.txt')

if __name__ == "__main__":
    run_models(regression=False)
