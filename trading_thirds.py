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
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

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

    dataset = pd.read_csv("dataset_6_12_spy_alt_norm.csv")
    print "Data read successfully."

    #Cleanup data
    dataset = dataset[dataset.RET1 != 'None']
    dataset = dataset[dataset.RET1 != 'Fail']
    dataset = dataset[dataset.RET6 != 'None']
    dataset = dataset[dataset.RET6 != 'Fail']
    dataset = dataset[dataset.RET12 != 'None']
    dataset = dataset[dataset.RET12 != 'Fail']
    dataset.dropna(axis=0, inplace=True)
    #dataset = dataset.drop(['COMPANY','NBC1','NBC2','NBC3','NBC4','NBC5','NBC6','NBV1','NBV2','NBV3','NBV4','NBV5','NBV6'], axis=1)
    dataset = dataset.drop(['COMPANY'], axis=1)
    dataset = dataset.drop(['MONTH'], axis=1)
    dataset = dataset.drop(['RET6'], axis=1)
    dataset = dataset.drop(['RET1'], axis=1)
    dataset.rename(index=str, columns={'RET12': 'RET'}, inplace=True)
    #dataset = dataset.drop(['MKTCAP'], axis=1)
    #dataset = dataset.drop(['SECTOR'], axis=1)
    dataset['MKTCAP'].replace('.*Small.', 'small', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Mid.', 'mid', inplace=True, regex=True)
    dataset['MKTCAP'].replace('.*Large.', 'large', inplace=True, regex=True)
    ret_data = dataset['RET'].apply(pd.to_numeric, errors='coerce')

    #eliminates points with many starting zeroes
    #dataset = dataset[dataset.apply(lambda x: count_zeroes(x) < 4,axis=1)]

    # TODO: Set the number of training points
    num_train = 80000

    #Print dataset statistics
    
    min_return = np.min(ret_data)
    max_return = np.max(ret_data)
    mean_return = np.mean(ret_data)
    median_return = np.median(ret_data)
    std_return = np.std(ret_data)
    first_third = np.percentile(ret_data,33)
    second_third = np.percentile(ret_data,66)

    print "Min return: " + str(min_return) + '\n'
    print "Max return: " + str(max_return) + '\n'
    print "Mean return: " + str(mean_return) + '\n'
    print "Median return: " + str(median_return) + '\n'
    print "Std Dev of return: " + str(std_return) + '\n'
    print "First third: " + str(first_third) + '\n'
    print "Second third: " + str(second_third) + '\n'

    if regression == False:
        #Convert return data into categories
        dataset.loc[(dataset.RET.astype(np.float64) > first_third) & (dataset.RET.astype(np.float64) < second_third), 'RETURN_CAT'] = np.str('middle')
        dataset.loc[(dataset.RET.astype(np.float64) >= second_third), 'RETURN_CAT'] = np.str('above')
        dataset.loc[(dataset.RET.astype(np.float64) <= first_third), 'RETURN_CAT'] = np.str('below')

        #dataset.loc[(dataset.RETURN.astype(np.float64) > -0.07) & (dataset.RETURN.astype(np.float64) < 0.07), 'RETURN_CAT'] = np.str('within_seven')
        #dataset.loc[dataset.RETURN.astype(np.float64) >= 0.07, 'RETURN_CAT'] = np.str('plus_seven')
        #dataset.loc[dataset.RETURN.astype(np.float64) <= -0.07, 'RETURN_CAT'] = np.str('minus_seven')

        print "\nNumber of stocks in middle: "
        print dataset[dataset['RETURN_CAT']=='middle'].count()[0]
        print "\nNumber of stocks above: "
        print dataset[dataset['RETURN_CAT']=='above'].count()[0]
        print "\nNumber of stocks below: "
        print dataset[dataset['RETURN_CAT']=='below'].count()[0]




    #Extract the feature and target columns
    feature_cols = list(dataset.columns[:-2])
    target_col = dataset.columns[-1]

    #Show the list of columns
    print "Feature columns:\n{}".format(feature_cols)
    print "\nTarget column: {}".format(target_col)

    #Separate data into feature data and target data
    X_all = dataset[feature_cols]
    y_all = dataset[target_col]

    X_all = preprocess_features(X_all)
    print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))



    # Set the number of testing points
    num_test = X_all.shape[0] - num_train

    # TODO: Shuffle and split the dataset into the number of training and testing points above
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = num_test, random_state = 1)

    # Show the results of the split
    #print "Training set has {} samples.".format(X_train.shape[0])
    #print "Testing set has {} samples.".format(X_test.shape[0])

    # Generate weights for Naive Bayes samples
    # y_weights = pd.Series(y_all.copy(deep=True))
    # y_weights.loc[y_weights == 'plus_ten'] = 4
    # y_weights.loc[y_weights == 'minus_ten'] = 4
    # y_weights.loc[y_weights == 'plus_five'] = 1
    # y_weights.loc[y_weights == 'minus_five'] = 1
    # y_weights.loc[y_weights == 'plus_five_ten'] = 4
    # y_weights.loc[y_weights == 'minus_five_ten'] = 4

    params={'C': [100]}
    #fit_params = {'sample_weight': y_weights}
    #fit_params = {'sample_weight': y_weights}
    clf_A = naive_bayes.GaussianNB()
    #clf_A = svm.SVC(C=10, random_state = 2, cache_size=1000)
    #clf_A = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3, weights='distance')
    #clf_B = AdaBoostClassifier(base_estimator=clf_svm, random_state=2)
    #clf_C = RandomForestClassifier(random_state = 2)
    #clf_gp = GaussianProcessClassifier(random_state = 2)

    reg_A = svm.SVR(cache_size=1000)

    start = time()
    #best_param, best_estimator = train_classifierGS(clf_B, X_all, y_all, params)
    #predict_labels(best_estimator, X_test, y_test)

    if regression:
        train_reg(reg_A,X_train,y_train,X_test,y_test)
    else:
        train_classifier(clf_A,X_train,y_train,X_test,y_test)

    end = time()
    print "Trained in {:.4f} seconds.".format(end - start)
    joblib.dump(reg_A, 'nb_reg.pkl')

if __name__ == "__main__":
    run_models(regression=True)
