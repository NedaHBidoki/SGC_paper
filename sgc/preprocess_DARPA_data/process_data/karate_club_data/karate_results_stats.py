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

db = 'karate'
subject = 'karate'
#subject = 'hashtags_2'
#subject = 'sensitivity'
#subject = 'polarity/polarity_wo_hashtags'
#subject = 'k_top_hashtags'

base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/pytorch/SGC-master/data/sgc_results/%s/'%(db)
save_dir='/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/process_data/karate_club_data/plots/%s/'%(db)
 
sub_dirs = ['F1','confM']
for s in sub_dirs:
	if not os.path.exists(save_dir+s):
		os.makedirs(save_dir+s)
fls = [ f for f in os.listdir(base)]
df = pd.DataFrame()
columns = ['subject', 'fpr','tpr','threshold','F1','roc_auc','acc','prc_mac', 'prc_mic', 'prc_wei','time', 'TN', 'FP', 'FN', 'TP', 'k','epoch','bin']

for f in fls:
	df1 = pd.read_csv(base + f, names=columns, sep ='\t')
	df = pd.concat([df,df1],axis = 0)
print(set(df['k'].values))
df['subject'] = df['subject'].str.replace('hashtags.keyword_', '')

#### plotting top hashtags and degrees #####
def plot_top_hashtags(df):
	top_hastags =['infosec','cybersecurity','security','hacking','securityaffairs','cve','linux','exploit','0day','vulnerability']
	df_t_h = df[df['hashtag_col'].isin(top_hastags)]
	df_t_h.groupby(['subject']).mean()['F1'].plot.bar(yerr=df_t_h.groupby(['subject']).std(), color='grey', grid=True)
	plt.xticks(rotation='vertical')
	plt.xlabel('Hashtags')
	plt.ylabel('F1')
	plt.title('F1 values of top hashtags in Twitter/cve network')
	plt.legend()
	plt.tight_layout()
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
def plot_conf_max_k(df):
	for k in [2,4,6,8,10]:
		for epoch in [10,20,30,40,50,100,200]:
				df1 = df[(df['k']==k) & (df['epoch']==epoch) ]
				#df1  = df1.drop(['k'], axis=1)
				bp2= df1[['TN', 'FP', 'FN', 'TP']].boxplot(grid=True,fontsize=11, showfliers=False, widths=(0.1, 0.1, 0.1, 0.1), patch_artist=True)
				#plt.show()
				for box in bp2['boxes']:
					box.set(color='b', linewidth=2)
					box.set(hatch = '*')
					box.set(facecolor = 'y' )
				plt.xlabel('Metric')
				plt.ylabel('Count')
				plt.title('Confusion Matrix for k = %s'%k)
				#plt.legend()
				plt.savefig('%s/confM/k%s_epoch%s_conf_mtx.pdf'%(save_dir,k,epoch))
				plt.close()

#### plotting F1 vs epoch  #####
def plot_f1_epoch(df):
	print(len(df))
	print(df['bin'].max())
	for k in [2,4,6,8,10]:
			df1 = df[(df['k']==k)]
			print(len(df1))
			df1.groupby(['epoch']).mean()['F1'].plot.bar(yerr=df1.groupby(['epoch']).std(), edgecolor='b', color='y', grid=True, hatch="*")
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
			plt.xlabel('Epoch')
			plt.ylabel('F1')
			plt.title('F1 values of epoch in Twitter/cve network (k = %s)'%k)
			#plt.legend()
			plt.tight_layout()
			plt.savefig('%s/F1/f1_epoch_k%s.pdf'%(save_dir,k))
			plt.close()

			
#### plotting F1 vs epoch  #####
def plot_f1_k(df):
	print(len(df))
	for epoch in [10,20,30,40,50,100,200]:
			df1 = df[(df['epoch']==epoch)]
			print(len(df1))
			df1.groupby(['k']).mean()['F1'].plot.bar(yerr=df1.groupby(['k']).std(), width=0.2, edgecolor='b', color='y', grid=True, hatch="*")
			'''
			#### plotting 3D
			threedee = plt.figure().gca(projection='3d')
			threedee.scatter(df1['F1'], df1['bin'], df1['epoch'])
			threedee.set_xlabel('F1')
			threedee.set_ylabel('bin')
			threedee.set_zlabel('epoch')
			'''
			plt.xticks(rotation='vertical')
			plt.xlabel('K')
			plt.ylabel('F1')
			plt.title('F1 values of degree in Twitter/cve network (epoch = %s)'%epoch)
			#plt.legend()
			plt.tight_layout()
			plt.savefig('%s/F1/f1_k_epoch%s.pdf'%(save_dir,epoch))
			plt.close()
			
plot_conf_max_k(df)
plot_f1_epoch(df)
plot_f1_k(df)

#print(df.mean())	
#dd=pd.melt(df,id_vars=['k'],value_vars=['f1'],var_name='f1')
#sns.boxplot(y='k',x='value',data=dd,orient="h",hue='f1')
