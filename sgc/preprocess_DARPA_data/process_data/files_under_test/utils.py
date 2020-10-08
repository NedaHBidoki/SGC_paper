import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from itertools import compress

def feature_analysis(df, pr_col,k):
	print('Feature Engineering ....')
	features_df = 	df[[c for c in df.columns if c!= pr_col]]
	print('Number of users: ',len(df))
	print('Number of features:',len(list(df.columns)))
	selected_features = lre_feature_selection(features_df.values,df[pr_col].values,features_df.columns,k)
	return df[selected_features+[pr_col]], selected_features



def lre_feature_selection(X,y, columns, k):	
	print('LogisticRegression Feature Engineering ....')
	# LogisticRegression
	model = LogisticRegression()
	rfe = RFE(model, k)
	fit = rfe.fit(X, y)
	#print("Num Features: %s" % (fit.n_features_))
	#print("Selected Features: %s" % (fit.support_))
	#print("Feature Ranking: %s" % (fit.ranking_))
	#print('features:',X.shape,'len columns:',len(columns),'len selected features:',len(fit.support_))
	selected_features =  list(compress(columns, fit.support_))
	#print(selected_features)
	return selected_features

def find_correlated_features():
	corrmat = df.corr()
	top_corr_features = corrmat.index
	plt.figure(figsize=(20,20))
	#plot heat map
	g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
	plt.savefig('corr_test.png')

	#Correlation with output variable
	cor_target = abs(corrmat[pr_col])

	#Selecting highly correlated features
	relevant_features = cor_target[cor_target>0.5]
	print('relevant_features: ', relevant_features)
	plt.close()
	return relevant_features 

def SelectKBest_feature_selection(X,y):
	# SelectKBest
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X, y)
	X_train_fs = fs.transform(X)
	# what are scores for the features
	for i in range(len(fs.scores_)):
		print('Feature %d: %f' % (i, fs.scores_[i]))
	# plot the scores
	plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
	plt.savefig('bar.png')
	plt.show()
	
def ExtraTreesClassifier_feature_selection(X,y):
	# Tree-based feature selection
	clf = ExtraTreesClassifier(n_estimators=50)
	clf = clf.fit(X, y)
	clf.feature_importances_  
	model = SelectFromModel(clf, prefit=True)
	X_new_etc = model.transform(X)
	
	print('X_new_etc: ',X_new_etc.shape)

def VarianceThreshold_feature_selection(X,y):
	# Removing features with low variance
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X_new_var = sel.fit_transform(X)
	print('X_new_var:',X_new_var.shape)
	
	# L1-based feature selection
	lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
	model = SelectFromModel(lsvc, prefit=True)
	X_new_lsvc = model.transform(X)
	print('X_new_lsvc: ',X_new_lsvc.shape)

	
	return 


def test():
	X = [[1,0,1],[1,0,0],[0,0,0],[1,1,0]]
	y = [1,2,3,4]
	feature_analysis(X,y)
