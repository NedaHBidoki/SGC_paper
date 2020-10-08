import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os
from collections import Counter
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

db = 'tw_cve'
#subject = 'hashtags_2'
#subject = 'subjectivity/communities/hashtags/test/'
#subject = 'polarity/whole_network/polarity_w_hashtags/whole_network/'
#subject = 'polarity/communities/topics/test/'
#subject = 'polarity/communities/hashtags/test/'
subject = 'polarity/communities/hashtags/selected/'
subject = 'polarity/communities/topics/selected/'
#subject = 'polarity/polarity_wo_hashtags'
#subject = 'polarity/polarity_w_hashtags/topics'
#subject = 'polarity/polarity_w_hashtags/subset_of_hashtags/25'
#subject = 'polarity/polarity_w_hashtags/communities/epoc2000-4000'
#subject = 'k_top_hashtags'
base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/pytorch/SGC-master/data/sgc_results/%s/%s/'%(db,subject)
base_save_dir = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/process_data/plots/%s/%s/'%(db,subject)
sub_dirs = ['F1','confM']
'''
for s in sub_dirs:
	if not os.path.exists(save_dir+s):
		os.makedirs(save_dir+s)
'''
fls = [ f for f in os.listdir(base) if os.path.isfile(os.path.join(base,f))]
df = pd.DataFrame()
columns = ['subject', 'fpr','tpr','threshold','F1','roc_auc','acc','prc_mac', 'prc_mic', 'prc_wei','time', 'TN', 'FP', 'FN', 'TP', 'k','epoch','bin']

'''
for f in fls:
	df1 = pd.read_csv(base + f, names=columns, sep ='\t')
	df = pd.concat([df,df1],axis = 0)
print(set(df['k'].values))
df['subject'] = df['subject'].str.replace('hashtags.keyword_', '')
print(df.head())
'''
#### plotting top hashtags and degrees #####
def plot_top_hashtags(df):
	top_hastags =['infosec','cybersecurity','security','hacking','securityaffairs','cve','linux','exploit','0day','vulnerability']
	df_t_h = df[df['hashtag_col'].isin(top_hastags)]
	df_t_h.groupby(['subject']).mean()['F1'].plot.bar(yerr=df_t_h.groupby(['subject']).std(), color='grey', grid=True)
	plt.xticks(rotation='vertical')
	plt.xlabel('Hashtags',fontsize=11)
	plt.ylabel('F1',fontsize=11)
	#plt.title('F1 values of top hashtags in Twitter/cve network',fontsize=11)
	plt.legend()
	plt.tight_layout(True)
	plt.savefig('plots/%s/top_hashtags_f1.pdf'%(db))
	plt.show()
	sys.exit()

#### plotting hashtags and degrees #####
def plot_hashtags_k(df):
	df.groupby(['subject','k']).mean()['F1'].unstack().plot()
	plt.xticks(rotation='vertical')
	plt.show()
	sys.exit()

#### plotting conf. max.  #####
bins = [0]
def plot_conf_max_k(df,save_dir,file_):
	for k in [2,4,6,8,10]:
		for epoch in [600, 800, 1000,10000, 50000]:
			for b in bins:
				df1 = df[(df['k']==k) & (df['bin']==b)  & (df['epoch']==epoch) ]
				#df1  = df1.drop(['k'], axis=1)
				if len(df1)==0:
					continue
				bp1 = df1[['TN', 'FP', 'FN', 'TP']].boxplot(grid=True,fontsize=11, showfliers=False, widths=(0.1, 0.1, 0.1, 0.1), patch_artist=True #, notch=True # vertical box alignment
                         )
				#bp1 = plt.boxplot(df1[['TN', 'FP', 'FN', 'TP']], showfliers=False,  patch_artist=True)#
				for _, line_list in bp1.items():
				    for line in line_list:
					line.set_color('b')

				#plt.show()
				#print(type(bp1))
				#print(bp1)
				#sys.exit()
				for box in bp1['boxes']:
					box.set(color='b', linewidth=2)
					box.set(hatch = '*')
				        box.set(facecolor = 'y' )
				#plt.show()
				plt.xlabel('Metric',fontsize=11)
				plt.ylabel('Count',fontsize=11)
				#plt.title('Community size: %s, Net size: %s, Confusion Matrix for k = %s'%(file_.split('_')[7][4:],file_.split('_')[8][7:],k))
				#plt.legend()
				plt.tight_layout(True)
				if df1['TP'].mean()>0 and (df1['TN'].mean()>0 or df1['FN'].mean()>0):
					plt.savefig('%s/confM/***************k%s_epoch%s_b%s_%s_%s_conf_mtx.pdf'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:]))
					print('************** found one %s **************'%file_)

				elif df1['TN'].mean()>0 and (df1['TP'].mean()>0 or df1['FP'].mean()>0):
					plt.savefig('%s/confM/***************k%s_epoch%s_b%s_%s_%s_conf_mtx.pdf'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:]))
					print('************** found one: %s **************'%file_)
				else:
					plt.savefig('%s/confM/k%s_epoch%s_b%s_%s_%s_conf_mtx.pdf'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:]))
				plt.close()

#### plotting F1 vs epoch  #####
def plot_f1_epoch(df,save_dir,file_):
	#print(len(df))
	#print(df['bin'].max())
	for k in [2,4,6,8,10]:
		for b in bins:			
			df1 = df[(df['k']==k) & (df['bin']==b)]
			print(len(df1))
			if len(df1)==0:
				continue
			df1.groupby(['epoch']).mean()['F1'].plot.bar(yerr=df1.groupby(['epoch']).std(), edgecolor='b', color='y', grid=True, hatch="*",width=0.3, linewidth=3)
			'''
			#### plotting 3D
			threedee = plt.figure().gca(projection='3d')
			threedee.scatter(df1['F1'], df1['bin'], df1['epoch'])
			threedee.set_xlabel('F1')
			threedee.set_ylabel('bin')
			threedee.set_zlabel('epoch')
			'''
			#plt.show()
			plt.xticks(rotation='vertical')
			plt.xlabel('Epoch',fontsize=11)
			plt.ylabel('F1',fontsize=11)
			##plt.title('F1 values of epoch in Twitter/cve network (k = %s), Community size: %s'%(k,file_.split('_')[7][4:]))
			#plt.legend()
			plt.tight_layout(True)
			plt.savefig('%s/F1/f1_epoch_k%s_bin%s.pdf'%(save_dir,k,b))
			plt.close()

			
#### plotting F1 vs epoch  #####
def plot_f1_k(df,save_dir, file_):
	#print(len(df))
	print(df['epoch'].values)
	for epoch in set(df['epoch'].values.tolist()):
		for b in set(df['bin'].values.tolist()):			
			df1 = df[(df['epoch']==epoch) & (df['bin']==b)]
			#print(len(df1))
			if len(df1)==0:
				continue
			#df1.groupby(['k']).mean()['F1'].plot.bar(yerr=df1.groupby(['k']).std(), color='grey', grid=True)
			df1 = df1.groupby(['k'])['F1'].mean()
			print(df1.head())
			df1.plot.bar(edgecolor='b', color='y', grid=True, hatch="*",width=0.3, linewidth=3)
			'''
			#### plotting 3D
			threedee = plt.figure().gca(projection='3d')
			threedee.scatter(df1['F1'], df1['bin'], df1['epoch'])
			threedee.set_xlabel('F1')
			threedee.set_ylabel('bin')
			threedee.set_zlabel('epoch')
			'''
			plt.xticks(rotation='vertical')
			plt.xlabel('K',fontsize=11)
			plt.ylabel('F1',fontsize=11)
			plt.tight_layout(True)
			##plt.title('F1 values of degree in Twitter/cve network (epoch = %s), Community size: %s'%(epoch,file_.split('_')[7][4:]))
			#plt.legend()
			#plt.tight_layout()
			plt.savefig('%s/F1/f1_k_epoch%s_bin%s.pdf'%(save_dir,epoch,b))
			#plt.show()
			plt.close()
for f in fls:
	df = pd.read_csv(base + f, names=columns, sep ='\t')	
	print('file:',f,'len: ',len(df))
	save_dir = base_save_dir + ' '.join([str(elem) for elem in f.split('_')[3:9]]) 
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for s in sub_dirs:
		if not os.path.exists(os.path.join(save_dir,s)):
			os.makedirs(os.path.join(save_dir,s))	
	plot_f1_k(df,save_dir,f)
	#plot_f1_epoch(df,save_dir,f)
	plot_conf_max_k(df,save_dir,f)	

#print(df.mean())	
#dd=pd.melt(df,id_vars=['k'],value_vars=['f1'],var_name='f1')
#sns.boxplot(y='k',x='value',data=dd,orient="h",hue='f1')
