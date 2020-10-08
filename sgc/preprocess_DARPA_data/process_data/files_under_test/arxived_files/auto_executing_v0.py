import os 
import sys
import shutil 
import random
import time
import pandas as pd
db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn'
path1 = 'preprocess_DARPA_data/process_data'
path2 = 'pytorch/SGC-master/data'


def one_complete_run(prediction_col_n,k):
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
	os.system('python3 %s/%s/prepare_matrices.py %s'%(base,path1,prediction_col_n))


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

	print('Execcuting SGC reddit.py ...')
	#if prediction_col_n != 'polarity':
	os.system('python3 reddit.py --pr_col %s --degree %s' %(prediction_col_n,k))
	#else:
	#os.system('python3 reddit.py --pr_col polarity')


def main():
	db = 'twitter_cve'
	top_hashtags_file = 'process_data/predicted_hashtags/%s/top_hashtags/50_largest_hashtags.csv'%db
	top_hashtags = pd.read_csv(top_hashtags_file, sep=',',header = None,names=['hashtags','count'])['hashtags'].values
	start_time = time.time()
	n_iteration = 5


	pr_col_log = base + '/'+ path1 + '/predicted _top_hashtags.csv'
	if os.path.exists(pr_col_log):
		os.remove(pr_col_log)
	results_log = base + '/'+ path2 + '/top_hashtag_results.csv'
	if os.path.exists(results_log):
		os.remove(results_log)
	for k in [2,3,4,5,10]:
		for h in top_hashtags:
			for i in range(0,n_iteration):
				print('************** iteration %s ***********'%i)
				one_complete_run(h,k)


	'''
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
	'''
	for i in range(0,n_iteration):
		print('************** iteration %s ***********'%i)
		one_complete_run('polarity')
	'''
	print("--- %s seconds ---" % (time.time() - start_time))

main()

