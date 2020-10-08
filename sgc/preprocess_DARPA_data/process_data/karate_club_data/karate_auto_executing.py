import os 
import sys
import shutil 
import random
import time
import pandas as pd
#sys.path.append('/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/process_data/')
from test_karate_club_data import get_ordered_features_and_adjM_matrices
from test_karate_club_data import get_features_adjM_data

db = 'karate_club'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn'
path1 = 'preprocess_DARPA_data/process_data'
path2 = 'pytorch/SGC-master/data'


def one_complete_run(kibana_matrix,A,subject,pr_col,k,epo,b_inx):
	print('Changing directory to prepare new data...')
	os.chdir(base +'/'+ path1)	 
	#execfile('prepare_matrices.py')

	print('Removing old data ...')
	feat_matrix = '%s/%s/npz_data/%s/reddit_adj.npz'%(base,path1,db)
	adj_matrix = '%s/%s/npz_data/%s/reddit.npz'%(base,path1,db)
	if os.path.exists(feat_matrix):
		os.remove(feat_matrix)
	if os.path.exists(adj_matrix):
		os.remove(adj_matrix)

	print('Execcuting prepare matrix codes ...')
	#os.system('python3 %s/%s/prepare_matrices.py %s\t%s\t%s'%(base,path1,pr_col,kibana_matrix,A))
	get_features_adjM_data()

	print('Renaming new data...')
	os.rename('%s/%s/npz_data/%s/%s.npz'%(base,path1,db,db),'%s/%s/npz_data/%s/reddit.npz'%(base,path1,db))
	os.rename('%s/%s/npz_data/%s/%s_adj.npz'%(base,path1,db,db),'%s/%s/npz_data/%s/reddit_adj.npz'%(base,path1,db))


	print('Changing directory to SGC...')
	os.chdir(base + '/'+ path2)

	print('Removing old data ...')
	feat_matrix = 'reddit_adj.npz'
	adj_matrix = 'reddit.npz'
	if os.path.exists(feat_matrix):
		os.remove(feat_matrix)
	if os.path.exists(adj_matrix):
		os.remove(adj_matrix)

	print('Moving new data...')
	shutil.move('%s/%s/npz_data/%s/reddit_adj.npz'%(base,path1,db), '%s/%s/'%(base,path2))
	shutil.move('%s/%s/npz_data/%s/reddit.npz'%(base,path1,db), '%s/%s/'%(base,path2))


	print('Changing directory to SGC...')
	os.chdir(base + '/'+ 'pytorch/SGC-master/')

	print('Executing SGC reddit.py ...')
	#if pr_col != 'polarity':
	os.system('python3 reddit.py --pr_col %s --degree %s --subject %s --epochs %s --binx %s' %(pr_col,k,subject,epo,b_inx))
	#else:
	#os.system('python3 reddit.py --pr_col polarity')


def main():
	db = 'karate_club'
	dir_ = '/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/'
	#top_hashtags_file = dir_ + 'process_data/predicted_hashtags/%s/top_hashtags/50_largest_hashtags.csv'%db
	#top_hashtags = pd.read_csv(top_hashtags_file, sep=',',header = None,names=['hashtags','count'])['hashtags'].values[0:10]
	start_time = time.time()
	n_iteration = 5
	bins_set = [[-1,0,1]]
	for b_inx,bins in enumerate(bins_set): 
		kibana_matrix,A = get_ordered_features_and_adjM_matrices()
		'''
		############# predicting top k hashtags over degree (k)
		pr_col_log = base + '/'+ path1 + '/k_predicted _top_hashtags.csv'
		if os.path.exists(pr_col_log):
			os.remove(pr_col_log)
		results_log = base + '/'+ path2 + '/k_top_hashtag_results.csv'
		if os.path.exists(results_log):
			os.remove(results_log)
		print('getting feature matrix and adj matrix...')
		for h in top_hashtags:
			for k in [2,4,6, 8,10]:
				for i in range(0,n_iteration):
					print('************** iteration %s ***********'%i)
					one_complete_run(kibana_matrix,A,'test',h,k,200,b_inx)

		'''
		'''
		############# predicting random hashtags
		for s in [1,50,100,150,200,250,300,350,400,450,500,550]:
			start = s
			end = start + 5
			col_list = [i for i in range(start,end)]
			pr_col_log = base + '/'+ path1 + '/predicted hashtags.csv'
			if os.path.exists(pr_col_log):
				os.remove(pr_col_log)
			results_log = base + '/'+ path2 + '/hashtag_results.csv'
			if os.path.exists(results_log):
				os.remove(results_log)
			for c in col_list:
				for i in range(0,n_iteration):
					print('************** iteration %s ***********'%i)
					one_complete_run(c)

			print('Renaming results file...')
			os.rename('%s/%s/hashtag_results.csv'%(base,path2),'%s/%s/hashtag_results_%s_%s.csv'%(base,path2,start,end))
			os.rename('%s/%s/predicted hashtags.csv'%(base,path1),'%s/%s/predicted hashtags_%s_%s.csv'%(base,path1,start,end))
		'''
		
		############# predicting polarity 
		pr_col , subject = 'label', 'karate'
		results_log = base + '/'+ path2 + '/%s_results.csv'%subject
		if os.path.exists(results_log):
			os.remove(results_log)
		for epo in [10,20,30,40,50,100,200]:
			for k in [2,4,6, 8,10]:
				for i in range(0,n_iteration):
					print('************** iteration %s ***********'%i)
					one_complete_run(kibana_matrix,A,subject,pr_col,k,epo,b_inx)
		
	print("--- %s seconds ---" % (time.time() - start_time))

main()

