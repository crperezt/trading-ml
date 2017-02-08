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
from sklearn.ensemble import GradientBoostingRegressor

def preprocess_features(X):
	output = pd.DataFrame(index = X.index)
	for col, col_data in X.iteritems():
		if col_data.dtype == object:
			col_data = pd.get_dummies(col_data, prefix = col)
		output = output.join(col_data)

	return output

def performance_metric(y_true, y_pred):
	return r2_score(y_true,y_pred)

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train) 
    scoring_fnc = make_scorer(performance_metric)
    predict_labels(clf,X_test, y_test)

def train_classifierGS(clf, X_train, y_train, params=None):
    cv_iters = 2
    cv_sets = ShuffleSplit(X_train.shape[0], n_iter=cv_iters, test_size=0.20, random_state=0)
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object

    grid = GridSearchCV(clf, params, scoring=scoring_fnc, cv=cv_sets)

    grid.fit(X_train, y_train)
    print "best_params_ for the optimal model are: {}.".format(str(grid.best_params_))
    return grid.best_params_, grid.best_estimator_
    #print "cv_results.['params'][grid.best_params_] for the optimal model are: {}.".format(str(grid.cv_results_['params'][grid.best_index_]))      

def train_classifierCV(clf, X_train, y_train, params=None, fit_params=None):
    ''' Fits a classifier to the training data. '''

    cv_iters = 10
    cv_sets = ShuffleSplit(X_train.shape[0], n_iter=cv_iters, test_size=0.20, random_state=0)
    scoring_fnc = make_scorer(performance_metric)  
    # Start the clock, train the classifier, then stop the clock
    start = time()
    scores_array = cross_val_score(clf, X_train, y=y_train, fit_params=fit_params, scoring=scoring_fnc, cv=cv_sets)
    #grid.fit(X_train, y_train)
    #clf.fit(X_train, y_train, sample_weight=weights)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)
    print "\nCross-Validation scores for " + str(cv_iters) + " iterations:"
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
    print "R2 Score: \n" + str(r2_score(target,y_pred))

def run_models():

    # Read student data
    #type_list = [('NBC' + str(i), np.float64) for i in range(1,13)]
    #type_list.extend([('NBV' + str(i), np.float64) for i in range(1,13)])
    #type_dict = dict(type_list)

    dataset = pd.read_csv("dataset.csv")
    print "Data read successfully."

    #Cleanup data
    dataset = dataset[dataset.RETURN != 'None']
    dataset = dataset[dataset.RETURN != 'Fail']
    dataset.dropna(axis=0, inplace=True)
    #dataset = dataset.drop(['COMPANY','NBC1','NBC2','NBC3','NBC4','NBC5','NBC6','NBV1','NBV2','NBV3','NBV4','NBV5','NBV6'], axis=1)
    dataset = dataset.drop(['COMPANY'], axis=1)
    dataset['MKTCAP'].replace('.*Small.', 'small', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Mid.', 'mid', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Large.', 'large', inplace=True, regex=True)

    dataset.loc[(dataset.RETURN.astype(np.float64) > 0.0) & (dataset.RETURN.astype(np.float64) < 0.05), 'RETURN_CAT'] = np.str('plus_five')
    dataset.loc[(dataset.RETURN.astype(np.float64) > -0.05) & (dataset.RETURN.astype(np.float64) <= 0.0), 'RETURN_CAT'] = np.str('minus_five')
    dataset.loc[(dataset.RETURN.astype(np.float64) >= 0.05) & (dataset.RETURN.astype(np.float64) < 0.10), 'RETURN_CAT'] = np.str('plus_five_ten')
    dataset.loc[(dataset.RETURN.astype(np.float64) <= -0.05) & (dataset.RETURN.astype(np.float64) > -0.10), 'RETURN_CAT'] = np.str('minus_five_ten')
    dataset.loc[dataset.RETURN.astype(np.float64) >= 0.10, 'RETURN_CAT'] = np.str('plus_ten')
    dataset.loc[dataset.RETURN.astype(np.float64) <= -0.10, 'RETURN_CAT'] = np.str('minus_ten')


    dataset.to_csv('full_data.csv')

    #print "printing return col!"
    #print dataset.loc[dataset.RETURN.dtype != np.float64].head()


    #Extract the feature and target columns
    feature_cols = list(dataset.columns[:-2])
    target_col = dataset.columns[-2]

    #Show the list of columns
    print "Feature columns:\n{}".format(feature_cols)
    print "\nTarget column: {}".format(target_col)

    #Separate data into feature data and target data
    X_all = dataset[feature_cols]
    y_all = dataset[target_col]

    X_all = preprocess_features(X_all)
    #print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))




    # TODO: Set the number of training points
    num_train = 40000

    # Set the number of testing points
    num_test = X_all.shape[0] - num_train

    # TODO: Shuffle and split the dataset into the number of training and testing points above
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = num_test, random_state = 1)


    #params={'C': [10, 100, 1000]}
    #fit_params = {'sample_weight': y_weights}
    #fit_params = {'sample_weight': y_weights}
    clf_A = GradientBoostingRegressor()
    #params = None
    params={'max_depth': [10, 100, 1000, 8000]}
    #clf_B = svm.SVR(cache_size=1000)
    #clf_C = RandomForestClassifier(random_state = 2)

    start = time()
    best_param, best_estimator = train_classifierGS(clf_A, X_train, y_train, params)
    predict_labels(best_estimator, X_test, y_test)
    #train_classifier(clf_B,X_train,y_train,X_test,y_test)
    end = time()
    print "Trained in {:.4f} seconds.".format(end - start)
    joblib.dump(best_estimator, 'reg_model_GBR.pkl')

if __name__ == "__main__":
    run_models()
