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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from random import *
import sklearn.metrics as metrics
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def initialization(data_file):
	db = 'twitter_cve'
	base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/raw_data/%s/'%db
	kibana_file = 'twitter_cve_topic_attr.csv' #attr_74074.csv'#'user_attr_matrix2.csv'
	#data = pd.read_csv(base + kibana_file)
	data = pd.read_csv(data_file).iloc[:,:-1]
	data['bin_label'] = [randint(0,1) for b in range(1,len(data)+1)]
	return data

def info(data):
	print('DATA INFORMATION:')
	print(data.info())
	print('#########################################')
	print('DATA DESCRIPTION:')
	print(data.describe())
	print('#########################################')

def get_X_y(data):
	cols = [c for c in data.columns if c !=  'id_h' and 'label' not in c]
	X, y = data[cols], data['bin_label']
	return X,y

def split_data(X,y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
	return X_train, X_test, y_train, y_test

def svc_train(X_train, X_test, y_train, y_test):
	print('#########################################')
	print('SVC CLASSIFICATION:')
	svclassifier = SVC(kernel='linear')
	svclassifier.fit(X_train, y_train)
	preds = svclassifier.predict(X_test)
	return preds

def nb_train(X_train, X_test, y_train, y_test):
	print('#########################################')
	print('NB CLASSIFICATION:')
	model = GaussianNB()
	model.fit(X_train,y_train)
	preds = model.predict(X_test)
	return preds

def logreg_train(X_train, X_test, y_train, y_test):
	print('#########################################')
	print('LOGREG CLASSIFICATION:')
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	preds = logreg.predict(X_test)
	return preds

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def F1(preds, labels):
    #preds = output.max(1)[1]
    #preds = preds.cpu().detach().numpy()
    #labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    try:
        fpr, tpr, threshold = roc(labels, preds)
        roc_auc = metrics.auc(fpr, tpr)
    except:
        fpr, tpr, threshold,roc_auc= 'nan','nan','nan','nan'
    return micro, macro, fpr, tpr, threshold,roc_auc

def acc(labels, preds):
	return metrics.accuracy_score(labels, preds)

def precision(labels, preds):
	prc_mac = metrics.precision_score(labels, preds, average='macro', zero_division=1)
	prc_mic = metrics.precision_score(labels, preds, average='micro', zero_division=1)
	prc_wei = metrics.precision_score(labels, preds, average='weighted', zero_division=1)
	return prc_mac, prc_mic, prc_wei

def conf_matrix(labels, preds):
	tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
	return tn, fp, fn, tp

def roc(y_test, preds,save_dir):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    #plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.plot(fpr, tpr, 'b')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #save_dir = 'plots/%s/%s'%(subject,predictor)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('%s/roc_%s.pdf'%(save_dir,time.time()))
    return fpr, tpr, threshold

def evaluate_benchmarks(X_train, X_test, y_train, y_test, pr_col, subject, predictor,save_dir):
	benchmarks = ['svc','nb','logreg']
	for b in benchmarks:
		preds = eval(b + '_train(X_train, X_test, y_train, y_test)')
		tn, fp, fn, tp = conf_matrix(y_test, preds)
		fpr, tpr, threshold = roc(y_test, preds,save_dir)
		prc_mac, prc_mic, prc_wei = precision(y_test, preds)
		f1, macro, fpr, tpr, threshold,roc_auc = F1(preds, y_test)
		acuracy = acc(y_test, preds)
		with open("cls_benchmarks_%s_%s_results.csv"%(subject,predictor), "a") as myfile:
    			myfile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(pr_col,fpr,tpr,threshold,f1,roc_auc,acuracy,prc_mac, prc_mic, prc_wei, tn, fp, fn, tp,b))
			
def test(data):
	X,y = get_X_y(data)
	X_train, X_test, y_train, y_test = split_data(X,y)
	pr_col, subject, predictor = 'test','test','test'
	evaluate_benchmarks(X_train, X_test, y_train, y_test, pr_col, subject, predictor)
