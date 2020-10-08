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
from pylab import plot, show, savefig, xlim, figure,axes,hold
db = 'tw_cve'
#subject = 'hashtags_2'
#subject = 'subjectivity/communities/hashtags/test/'
#subject = 'polarity/whole_network/polarity_w_hashtags/whole_network/'
#subject = 'polarity/communities/topics/test/'
#subject = 'polarity/communities/hashtags/test/'
subject = 'polarity/communities/hashtags/selected/'
subject2 = 'polarity/communities/topics/selected/'
#subject = 'polarity/polarity_wo_hashtags'
#subject = 'polarity/polarity_w_hashtags/topics'
#subject = 'polarity/polarity_w_hashtags/subset_of_hashtags/25'
#subject = 'polarity/polarity_w_hashtags/communities/epoc2000-4000'
#subject = 'k_top_hashtags'
base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/pytorch/SGC-master/data/sgc_results/%s/%s/'%(db,subject)
base2 = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/pytorch/SGC-master/data/sgc_results/%s/%s/'%(db,subject2)
base_save_dir = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/process_data/plots/%s/%s/Merged/'%(db,subject)
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
	plt.savefig('plots/%s/top_hashtags_f1.png'%(db))
	plt.show()
	sys.exit()

#### plotting hashtags and degrees #####
def plot_hashtags_k(df):
	df.groupby(['subject','k']).mean()['F1'].unstack().plot()
	plt.xticks(rotation='vertical')
	plt.show()
	sys.exit()


def test(data_a,data_b, name):
	ticks = ['TN', 'FP', 'FN', 'TP']

	def set_box_color(bp, color, c):
	    plt.setp(bp['boxes'], color=color)
	    plt.setp(bp['whiskers'], color='r')
	    plt.setp(bp['caps'], color='b')
	    plt.setp(bp['medians'], color=color)
            for box in bp['boxes']:
		box.set( linewidth=2)
		box.set(hatch = '*')
		box.set(facecolor = c )

	plt.figure()

	bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.3, patch_artist=True, showcaps=True,showbox =True)
	bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.3, patch_artist=True, showcaps=True,showbox =True)
	set_box_color(bpl, 'b','y') # colors are from http://colorbrewer2.org/
	set_box_color(bpr, 'y','b')

	# draw temporary red and blue lines and use them to create a legend
	plt.plot([], c='b', label='Hashtags')
	plt.plot([], c='y', label='Topics')
	plt.legend()

	plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
	plt.xlim(-2, len(ticks)*2)
	plt.ylim(0, 8)
	plt.grid(True)
	plt.xlabel('Metric',fontsize=13)
	plt.ylabel('Count',fontsize=13)
	plt.tight_layout()
	plt.savefig(name)
	#plt.show()
#### plotting conf. max.  #####
bins = [0]
def plot_conf_max_k(df,se_df, save_dir,file_):
	for k in [2,4,6,8,10]:
		for epoch in [600, 800, 1000,10000, 50000]:
			for b in bins:
				
				df1 = df[(df['k']==k) & (df['bin']==b)  & (df['epoch']==epoch) ]
				df2 = se_df[(se_df['k']==k) & (se_df['bin']==b)  & (se_df['epoch']==epoch) ]
				#df1  = df1.drop(['k'], axis=1)
				if len(df1)==0:
					continue
				name = '%s/confM/k%s_epoch%s_b%s_%s_%s_conf_mtx.png'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:])
				df1 = df1[['TN', 'FP', 'FN', 'TP']]
				df2 = df2[['TN', 'FP', 'FN', 'TP']]
				data_a = df1.transpose().values.tolist()
				data_b = df2.transpose().values.tolist()
				test(data_a,data_b,name)
				continue
				bp1 = df1[['TN', 'FP', 'FN', 'TP']].boxplot(grid=True,fontsize=11, showfliers=False, widths=(0.1, 0.1, 0.1, 0.1), patch_artist=True) #, notch=True # vertical box alignment )
				bp2 = df2[['TN', 'FP', 'FN', 'TP']].boxplot(grid=True,fontsize=11, showfliers=False, widths=(0.1, 0.1, 0.1, 0.1), patch_artist=True) #, notch=True # vertical box alignment )
				#bp1 = plt.boxplot(df1[['TN', 'FP', 'FN', 'TP']], showfliers=False,  patch_artist=True)#
				for _, line_list in bp1.items():
					for line in line_list:
						line.set_color('b')

				
				for box in bp1['boxes']:
					box.set(color='black', linewidth=2)
					box.set(hatch = '/')
					box.set(facecolor = 'y' )
				for box in bp2['boxes']:
					box.set(color='r', linewidth=2)
					box.set(hatch = '*')
					box.set(facecolor = 'y' )
				#plt.show()
				plt.xlabel('Metric',fontsize=11)
				plt.ylabel('Count',fontsize=11)
				#plt.title('Community size: %s, Net size: %s, Confusion Matrix for k = %s'%(file_.split('_')[7][4:],file_.split('_')[8][7:],k))
				#plt.legend()
				plt.tight_layout()
				if df1['TP'].mean()>0 and (df1['TN'].mean()>0 or df1['FN'].mean()>0):
					plt.savefig('%s/confM/***************k%s_epoch%s_b%s_%s_%s_conf_mtx.png'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:]))
					print('************** found one %s **************'%file_)

				elif df1['TN'].mean()>0 and (df1['TP'].mean()>0 or df1['FP'].mean()>0):
					plt.savefig('%s/confM/***************k%s_epoch%s_b%s_%s_%s_conf_mtx.png'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:]))
					print('************** found one: %s **************'%file_)
				else:
					plt.savefig('%s/confM/k%s_epoch%s_b%s_%s_%s_conf_mtx.png'%(save_dir,k,epoch,b,file_.split('_')[5]+file_.split('_')[6],file_.split('_')[7][4:]))
				plt.close()

#### plotting F1 vs epoch  #####
def plot_f1_epoch(df,se_df,save_dir,file_):
	#print(len(df))
	#print(df['bin'].max())
	for k in [2,4,6,8,10]:
		for b in bins:	
			fig, ax = plt.subplots()				
			df1 = df[(df['k']==k) & (df['bin']==b)]
			df2 = se_df[(se_df['k']==k) & (se_df['bin']==b)]
			print(len(df1))
			if len(df1)==0:
				continue
			df1.groupby(['epoch']).mean()['F1'].plot.bar(yerr=df1.groupby(['epoch']).std(), edgecolor='black', color='y', grid=True, hatch="*",width=0.2, linewidth=3,label='Hashtags',ax=ax,position=0)
			df2.groupby(['epoch']).mean()['F1'].plot.bar(yerr=df1.groupby(['epoch']).std(), edgecolor='y', color='black', grid=True, hatch="*",width=0.2, linewidth=3, label='Topics',ax=ax,position=1)
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
			plt.xlabel('Epoch',fontsize=12)
			plt.ylabel('F1',fontsize=12)
			##plt.title('F1 values of epoch in Twitter/cve network (k = %s), Community size: %s'%(k,file_.split('_')[7][4:]))
			plt.legend()
			plt.tight_layout(True)
			plt.savefig('%s/F1/f1_epoch_k%s_bin%s.png'%(save_dir,k,b))
			plt.close()

			
#### plotting F1 vs epoch  #####
def plot_f1_k(df,se_df, save_dir, file_):
	#print(len(df))
	print(df['epoch'].values)
	for epoch in set(df['epoch'].values.tolist()):
		for b in set(df['bin'].values.tolist()):	
			
			fig, ax = plt.subplots()		
			df1 = df[(df['epoch']==epoch) & (df['bin']==b) & (df['k'].isin([0,1,2]))]
			df2 = se_df[(se_df['epoch']==epoch) & (se_df['bin']==b) & (df['k'].isin([0,1,2]))]
			#print(len(df1))
			if len(df1)==0:
				continue
			#df1.groupby(['k']).mean()['F1'].plot.bar(yerr=df1.groupby(['k']).std(), color='grey', grid=True)
			df1 = df1.groupby(['k'])['F1'].mean()
			print(df1.head())
			df1.plot.bar(edgecolor='black', color='y', grid=True, hatch="*",width=0.2, linewidth=3, label='Hashtags',ax=ax,position=0)
			df2 = df2.groupby(['k'])['F1'].mean()
			df2.plot.bar(edgecolor='y', color='black', grid=True, hatch="*",width=0.2, linewidth=3, label='Topics',ax=ax,position=1)
			'''
			#### plotting 3D
			threedee = plt.figure().gca(projection='3d')
			threedee.scatter(df1['F1'], df1['bin'], df1['epoch'])
			threedee.set_xlabel('F1')
			threedee.set_ylabel('bin')
			threedee.set_zlabel('epoch')
			'''
			plt.xticks(rotation='vertical')
			plt.xlabel('K',fontsize=12)
			plt.ylabel('F1',fontsize=12)
			plt.tight_layout(True)
			##plt.title('F1 values of degree in Twitter/cve network (epoch = %s), Community size: %s'%(epoch,file_.split('_')[7][4:]))
			plt.legend()
			#plt.tight_layout()
			#plt.show()
			plt.savefig('%s/F1/f1_k_epoch%s_bin%s.png'%(save_dir,epoch,b))


			plt.close()

#fls2=[('polarity_w_hashtags_hashtags_40_community_14_size111_netSize59_results.csv','polarity_w_hashtags_topics_40_community_14_size111_netSize52_results.csv'),('polarity_w_hashtags_hashtags_40_community_11_size125_netSize37_results.csv','polarity_w_hashtags_topics_40_community_11_size125_netSize35_results.csv')]
fls2=[('polarity_w_hashtags_hashtags_600_community_3_size414_netSize105_results.csv','polarity_w_hashtags_topics_600_community_3_size414_netSize99_results.csv'),('polarity_w_hashtags_hashtags_600_community_11_size125_netSize37_results.csv','polarity_w_hashtags_topics_600_community_11_size125_netSize35_results.csv'), ('polarity_w_hashtags_hashtags_600_community_13_size113_netSize33_results.csv','polarity_w_hashtags_topics_600_community_13_size113_netSize29_results.csv')]

for f1,f2 in fls2:
	df = pd.read_csv(base + f1, names=columns, sep ='\t')
	df2 = pd.read_csv(base2 + f2, names=columns, sep ='\t')
	print('file:##############',f1,'len: ',len(df))
	save_dir = base_save_dir + ' '.join([str(elem) for elem in f1.split('_')[3:9]]) 
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for s in sub_dirs:
		if not os.path.exists(os.path.join(save_dir,s)):
			os.makedirs(os.path.join(save_dir,s))
	plot_conf_max_k(df,df2,save_dir,f1)
	plot_f1_epoch(df,df2,save_dir,f)
	plot_f1_k(df,df2,save_dir,f)	
	continue



#print(df.mean())	
#dd=pd.melt(df,id_vars=['k'],value_vars=['f1'],var_name='f1')
#sns.boxplot(y='k',x='value',data=dd,orient="h",hue='f1')
