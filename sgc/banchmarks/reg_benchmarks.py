import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/raw_data/%s/'%db
kibana_file = 'twitter_cve_topic_attr.csv' #attr_74074.csv'#'user_attr_matrix2.csv'
data = pd.read_csv(base + kibana_file)
data['bin_label'] = 1

def info(data):
	print('DATA INFORMATION:')
	print(data.info())
	print('#########################################')
	print('DATA DESCRIPTION:')
	print(data.describe())
	print('#########################################')

def get_X_y(data):
	cols = [c for c in data.columns if c !=  'id_h' and 'label' not in c]
	X, y = data[cols], data['extension.polarity: Descending']
	return X,y

def xgb_split_data(X,y):
	data_dmatrix = xgb.DMatrix(data=X,label=y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
	return X_train, X_test, y_train, y_test


def split_data(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
	return X_train, X_test, y_train, y_test

def xgb_train(X_train, X_test, y_train, y_test):
	print('#########################################')
	print('XGB TRAINING')
	xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
	xg_reg.fit(X_train,y_train)
	preds = xg_reg.predict(X_test)
	return preds

def rf_train(X_train, X_test, y_train, y_test):
	print('#########################################')
	print('RF TRAINING')
	# Instantiate model with 1000 decision trees
	rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
	rf.fit(X_train, y_train)
	preds = rf.predict(X_test)
	return preds

def rmse(y_test, preds):
	rmse = np.sqrt(mean_squared_error(y_test, preds))
	print("RMSE: %f" % (rmse))
	return rmse

def evaluate_benchmarks(X_train, X_test, y_train, y_test, pr_col, subject, predictor):
	benchmarks = ['xgb','rf']
	for b in benchmarks:
		preds = eval(b+'_train(X_train, X_test, y_train, y_test)')
		rms = rmse(y_test, preds)
		with open("reg_benchmarks_%s_%s_results.csv"%(subject,predictor), "a") as myfile:
    			myfile.write("%s\t%s\t%s\n"%(pr_col,rms,b))
			
X,y = get_X_y(data)
X_train, X_test, y_train, y_test = split_data(X,y)
pr_col, subject, predictor = 'test','test','test'
evaluate_benchmarks(X_train, X_test, y_train, y_test, pr_col, subject, predictor)
