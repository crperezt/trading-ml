# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

def preprocess_features(X):
	output = pd.DataFrame(index = X.index)
	for col, col_data in X.iteritems():
		if col_data.dtype == object:
			col_data = pd.get_dummies(col_data, prefix = col)
		output = output.join(col_data)

	return output

# Read student data
type_list = [('NBC' + str(i), np.float64) for i in range(1,13)]
type_list.extend([('NBV' + str(i), np.float64) for i in range(1,13)])
type_dict = dict(type_list)

dataset = pd.read_csv("dataset.csv", dtype=type_dict)
print "Data read successfully."

#Cleanup data
dataset = dataset[dataset.RETURN != 'None']
dataset = dataset[dataset.RETURN != 'Fail']
dataset = dataset.drop('COMPANY', axis=1)
dataset['MKTCAP'].replace('.*Small.', 'small', inplace=True, regex=True)
dataset['MKTCAP'].replace('.*Mid.', 'mid', inplace=True, regex=True)
dataset['MKTCAP'].replace('.*Large.', 'large', inplace=True, regex=True)

# dataset.loc[(dataset.RETURN.astype(float) < 0.05) & (dataset.RETURN.astype(float) > -0.05), 'RETURN_CAT'] = 'within5'
# dataset.loc[(dataset.RETURN.astype(float) >= 0.05) & (dataset.RETURN.astype(float) < 0.10), 'RETURN_CAT'] = 'plus5to10'
# dataset.loc[(dataset.RETURN.astype(float) <= -0.05) & (dataset.RETURN.astype(float) > -0.10), 'RETURN_CAT'] = 'minus5to10'
# dataset.loc[dataset.RETURN.astype(float) >= 0.10, 'RETURN_CAT'] = 'plus10'
# dataset.loc[dataset.RETURN.astype(float) <= -0.10, 'RETURN_CAT'] = 'minus10'

dataset.loc[(dataset.RETURN < 0.05) & (dataset.RETURN > -0.05), 'RETURN_CAT'] = 'within5'
dataset.loc[(dataset.RETURN >= 0.05) & (dataset.RETURN < 0.10), 'RETURN_CAT'] = 'plus5to10'
dataset.loc[(dataset.RETURN <= -0.05) & (dataset.RETURN > -0.10), 'RETURN_CAT'] = 'minus5to10'
dataset.loc[dataset.RETURN >= 0.10, 'RETURN_CAT'] = 'plus10'
dataset.loc[dataset.RETURN <= -0.10, 'RETURN_CAT'] = 'minus10'


#Extract the feature and target columns
feature_cols = list(dataset.columns[:-2])
target_col = dataset.columns[-1]

#Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

#Separate data into feature data and target data
X_all = dataset[feature_cols]
y_all = dataset[target_col]

print "\nFeature values:"
print X_all.head()

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))





#print pd.get_dummies(dataset.SECTOR).head()