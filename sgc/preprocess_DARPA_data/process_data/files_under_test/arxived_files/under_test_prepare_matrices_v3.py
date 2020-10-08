import pandas as pd
import sys
import random
import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os
sys.path.insert(0, '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/banchmarks')
from clsf_benchmarks import *
import utils

#print ("the prdicted column index %s" % (sys.argv[1]))
db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/raw_data/%s/'%db
#wiki_file = 'component_0.csv'#'network_twitter_cve.csv'
kibana_file = 'attr_74074.csv' #'twitter_cve_topic_attr.csv' # #'user_attr_matrix2.csv', 


def get_networks_nodes(wiki_file):
	df = pd.read_csv(base + wiki_file)
	nodes = df['node'].values.tolist()
	parents = df['parent'].values.tolist()
	nodes.extend(parents)
	return nodes


def get_kibana_nodes_matrix(kibana_file,bins, pr_col, predictors):
	#df = pd.read_csv(base + kibana_file)
	rm_quote = lambda x: x.replace('"', '')
	df = pd.read_csv(base + kibana_file, thousands=',')# nrows = 2000, skiprows=2000)
	df = df.fillna(0)
	df.columns = df.columns.str.replace(": Descending", "")
	df  = df.set_index('id_h')
	print(df['extension.polarity'].max())
	#@@@@polarity_category = pd.cut(df['extension.polarity'],bins=bins,labels=range(len(bins)-1))
	#@@@@df['polarity_category']= polarity_category
	df['polarity_category'] = (df['extension.polarity'] < 0).astype(int)
	df['polarity_category']= pd.to_numeric(	df['polarity_category'])
	print(df.loc[df['polarity_category']==0][['polarity_category','extension.polarity']])
	df = df.drop('extension.polarity', axis=1)
	df['possibly_sensitive'] = df['possibly_sensitive'].astype(int)
	#print('min:',df['polarity_category'].min(),'     max:', df['polarity_category'].max())
	if predictors == 'topics':
		column_names_for_onehot = ["topics"]
	#elif predictors == 'hashtags':
	#column_names_for_onehot = ["hashtags.keyword"]
	else:
		column_names_for_onehot = [ "hashtags.keyword"]#,"possibly_sensitive"] "user.time_zone.keyword",
	dum_df = pd.get_dummies(df,columns=column_names_for_onehot, dummy_na=False)
	max_cols =[ 'polarity_category', 'user.followers_count', 'extension.subjectivity', 'possibly_sensitive', 'user.friends_count']
	dictOffuncs = { i : 'count' for i in dum_df.columns if not i in max_cols}
	dictOfmax ={ i : 'max' for i in max_cols}
	dictOffuncs= {**dictOffuncs, **dictOfmax}
	dum_df = dum_df.groupby(dum_df.index).agg(dictOffuncs)
	#dum_df= dum_df.drop('Count')
	cols_to_norm = list(dum_df.columns)
	subject_cols = ['polarity_category', 'possibly_sensitive']
	for i in subject_cols: 
    		try: 
        		cols_to_norm.remove(i) 
    		except ValueError: 
        		pass
	#cols_to_norm.remove('polarity_category')
	#print((dum_df[max_cols].head()))
	#print(dum_df.dtypes)
	dum_df[cols_to_norm] = dum_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
	#print(dum_df[dum_df.columns[3]].head())
	#utils.feature_analysis(dum_df, pr_col)
	#dum_df = utils.feature_selection(dum_df[[c for c in dum_df.columns if ('hashtag' in c) and ('nan' not in c)]].values,dum_df['polarity_category'].values)
	#sys.exit()
	#print([c for c in dum_df.columns if 'sensiti' in c])
	#print(dum_df['possibly_sensitive_True'].head())
	#print(dum_df['user.followers_count'].head())
	#sys.exit()
	nodes = df.index
	#print('Subject head:', dum_df[subject_cols].head())
	#print('Normalized features head:', dum_df[cols_to_norm].head())
	return nodes, dum_df

def get_wiki_adj_matrix(wiki_file):
	print(' Generating Adjacency Matrix from %s...'%wiki_file)
	df = pd.read_csv(base + wiki_file)#, nrows=2000, skiprows=2000, names=['created_at','node','parent'])
	nodes = df['node'].values.tolist()
	parents = df['parent'].values.tolist()
	all_nodes = set(nodes + parents)
	edges = zip(nodes,parents)
	G = nx.Graph()
	G.add_edges_from(list(edges))
	A = nx.adjacency_matrix(G)
	A_pd = nx.to_pandas_adjacency(G, nodelist=all_nodes, dtype=int)
	#A_pd.to_csv(base + 'A_matrix.csv')
	print('A_pd.shape', A_pd.shape)
	return A_pd


def get_top_hashtags(kibana_matrix,n):
	cols = list(kibana_matrix.columns)
	keyword = 'hashtags'
	feat_cols = [c for c in cols if (keyword in c) and ('nan' not in c)]
	#print('m largest:',type(kibana_matrix[feat_cols].sum().nlargest(13, keep='first')))
	top_hashtags = kibana_matrix[feat_cols].sum().nlargest(n, keep='first')
	top_hashtags.to_csv('top_hashtags/%s_largest_hashtags.csv'%n)
	top_hashtags = top_hashtags.index.tolist()
	#top_hashtags_indexes = [feat_cols.index(c) for c in top_hashtags if c in feat_cols]
	#print('top_hashtags_indexes:',top_hashtags)
	#resultFile = open('top_hashtags/n_largest_hashtags_indexes.csv','w')
	#for r in top_hashtags_indexes:
	#    resultFile.write(r + "\n")
	#resultFile.close()
	return top_hashtags

def intersection_data(A_pd,kibana_matrix):
	A_pd = A_pd[A_pd.index.isin(kibana_matrix.index)]
	A_pd = A_pd[np.intersect1d(A_pd.columns, kibana_matrix.index)]
	kibana_matrix = kibana_matrix[kibana_matrix.index.isin(A_pd.index)]
	print('A_pd.shape', A_pd.shape)
	print('kibana_matrix.shape', kibana_matrix.shape)
	return A_pd, kibana_matrix

def get_ordered_features_and_adjM_matrices(bins, col , subject, predictors, top, wiki_file):
	A_pd = get_wiki_adj_matrix(wiki_file)
	kibana_nodes, kibana_matrix = get_kibana_nodes_matrix(kibana_file, bins, col, predictors)
	A_pd, kibana_matrix = intersection_data(A_pd,kibana_matrix)
	kibana_matrix, feat_cols, pr_cols = get_feat_prediction_cols(subject, predictors, col, kibana_matrix, top)
	if len(A_pd) == 0 or len(kibana_matrix)  == 0:
		return pd.DataFrame(), np.nan,np.nan,np.nan
	kibana_matrix = kibana_matrix.reindex(A_pd.index)
	#print('A_pd.index:', (A_pd).head())
	#print('kibana_matrix.index:', (kibana_matrix).head())
	#print('A_pd.index:', A_pd.index[:5])
	#print('kibana_matrix.index:', kibana_matrix.index[:5])
	print('%%%%%%%%%%%%%%%%%%%%%%%%%')
	print('Network file: ',wiki_file)
	print('Number of nodes in A: ',len(A_pd))
	print('Number of nodes in M:',len(kibana_matrix))
	print(A_pd.index.equals(kibana_matrix.index))
	#print(A_pd.index.difference(kibana_matrix.index))
	A_pd = A_pd.reset_index()
	kibana_matrix = kibana_matrix.reset_index()
	#print(A_pd['index'].head())
	#print(kibana_matrix['index'].head())
	A_pd['index'] = pd.Series(A_pd['index']).astype('category').cat.codes.values
	A_pd = A_pd.set_index('index')
	#print(A_pd.index[:5])
	kibana_matrix['index'] = pd.Series(kibana_matrix['index']).astype('category').cat.codes.values
	kibana_matrix = kibana_matrix.set_index('index')
	#print(A_pd.columns[:5])
	A_pd.columns=pd.Series(A_pd.columns).astype('category').cat.codes.values
	#print(A_pd.columns[:5])
	#print(kibana_matrix['index'].head())
	return kibana_matrix,A_pd.values,feat_cols, pr_cols

def save_npz_adj_matrix(M, name, dir_):
	sparse_matrix = scipy.sparse.csc_matrix(M)
	#print(sparse_matrix.shape)
	sparse_matrix.todense()
	scipy.sparse.save_npz(dir_ + '%s_adj.npz'%name, sparse_matrix)
	sparse_matrix = scipy.sparse.load_npz(dir_ + '%s_adj.npz'%name)

def draw_net(G):
	nx.draw(G, pos=nx.spring_layout(G))
	plt.show()

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    print('in train_validate_test_split...')
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return perm[:train_end], perm[train_end:validate_end], perm[validate_end:]

def customized_train_validate_test_split(pr_col, df, train_percent=.6, validate_percent=.2, seed=None):
    #colname = sys.argv[1]
    colname = pr_col
    print('colname:',colname)
    one_value_list = df.index[df[colname] == 1].tolist()
    if len(one_value_list)<3:
        sys.exit()
    one_tr,one_va, one_te =  np.array_split(np.array(one_value_list),3)
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    perm = np.setdiff1d(perm,np.array(one_value_list))
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    #train = df.iloc[perm[:train_end]]
    #validate = df.iloc[perm[train_end:validate_end]]
    #test = df.iloc[perm[validate_end:]]
    #return perm[:train_end], perm[train_end:validate_end], perm[validate_end:]
    print(type(perm[:train_end]))
    train_inx = np.concatenate((perm[:train_end], one_tr), axis=None) 
    val_inx = np.concatenate((perm[train_end:validate_end], one_va), axis=None)  
    test_inx = np.concatenate((perm[validate_end:], one_te), axis=None) 
    #print('one_tr:',one_tr)
    #print('one_va:',one_va)
    #print('one_te:',one_te)  
    #print('train_inx:',train_inx)
    #print('val_inx:',val_inx)
    #print('test_inx:',test_inx)

    #print('one_value_list:',len(one_value_list))  
    #print('perm:',len(perm))  
    #print('one_tr:',len(one_tr))
    #print('one_va:',len(one_va))
    #print('one_te:',len(one_te)) 
    #print('train_inx:',len(train_inx))
    #print('val_inx:',len(val_inx))
    #print('test_inx:',len(test_inx))
    return train_inx, val_inx, test_inx


def get_feat_prediction_cols(subject, predictors, col, kibana_matrix, top):
	cols = list(kibana_matrix.columns)
	if predictors == 'topics':
		keyword = 'topic'
	elif predictors == 'hashtags':
		keyword = 'hashtags'
	kibana_matrix.dropna(inplace=True)
	#kibana_matrix['user.followers_count'] = pd.to_numeric(kibana_matrix['user.followers_count'])
	#kibana_matrix['user.friends_count'] = pd.to_numeric(kibana_matrix['user.friends_count'])
	feat_cols = [c for c in cols if (keyword in c) and ('nan' not in c)]
	print('kibana_matrix:',len(kibana_matrix))
	#print(kibana_matrix['user.followers_count'].head())
	if len(kibana_matrix) ==0:
		return np.nan, np.nan, np.nan
	print(type(kibana_matrix['user.followers_count'].iloc[0]))

	if subject == 'polarity_wo_hashtags':
		print('subject is polarity w/o hashtags...')
		pr_cols = 'polarity_category'
		#pr_cols = 'extension.polarity'
		feat_cols = ['user.followers_count', 'user.friends_count']
	elif subject == 'polarity_w_hashtags':
		print('subject is polarity w hashtags...')
		pr_cols = 'polarity_category'
		#pr_cols = 'extension.polarity'
		feat_cols = feat_cols + ['user.followers_count', 'user.friends_count']
	elif subject == 'sensitivity':
		print('subject is sensitivity...')
		pr_cols= 'possibly_sensitive'
		feat_cols = [col for col in feat_cols if not subject in col] + ['user.followers_count', 'user.friends_count']
		#print(rm_cols)
		#if len(rm_cols)>0:
		#	feat_cols.remove(rm_cols)
	#elif int(sys.argv[1]) > len(feat_cols):
	#	print('index out of bound')
	#	sys.exit()
	else:
		#pr_cols_number = int(sys.argv[1]) #random.randrange(len(feat_cols))
		#pr_cols = feat_cols[pr_cols_number]
		print('subject is hashtags...')
		pr_cols = col
		if pr_cols in feat_cols:
			feat_cols.remove(pr_cols)
		with open("predicted hashtags.csv", "a") as myfile:
			#myfile.write("%s\t%s\n"%(pr_cols_number, pr_cols))
			myfile.write("%s\n"%(pr_cols))
	
	print('Number of predictors (hashtags): %s'%len(feat_cols))
	print('Target variable:',pr_cols)
	if keyword == 'hashtags':
		try:
			kibana_matrix, feat_cols = utils.feature_analysis(kibana_matrix[feat_cols+[pr_cols]],pr_cols, top)
		except:
			print('ERROR in FEATURE ENGINEEIRING')
			return pd.DataFrame(),np.nan,np.nan
	return kibana_matrix, feat_cols, pr_cols


def get_features_adjM_data(kibana_matrix,A,subject,feat_cols, pr_cols):
	#kibana_matrix,A = get_ordered_features_and_adjM_matrices()
	#kibana_matrix,A = sys.argv[2], sys.argv[3]
	#print(len(kibana_matrix))
	#print(kibana_matrix.sum())
	dir_ = 'npz_data/%s/'%db
	if not os.path.exists(dir_):
		os.makedirs(dir_)
	outfile = dir_ + '%s.npz'%db
	save_npz_adj_matrix(A, db, dir_)
	#feat_cols, pr_cols = get_feat_prediction_cols(subject, predictors, col,kibana_matrix)
	feats = kibana_matrix[feat_cols].values
	print(np.shape(feats))
	y = kibana_matrix[pr_cols]
	if subject == 'polarity_wo_hashtags' or subject == 'polarity_w_hashtags' or subject == 'sensitivity':
		print('regular data splitting...')
		train_index, val_index, test_index = train_validate_test_split(kibana_matrix, train_percent=.6, validate_percent=.2, seed=None)
	else:
		print('customized data splitting...')
		train_index, val_index, test_index = customized_train_validate_test_split(pr_cols,kibana_matrix, train_percent=.6, validate_percent=.2, seed=None)
	y_train, y_val, y_test = y.iloc[train_index].values.reshape(-1), y.iloc[val_index].values.reshape(-1), y.iloc[test_index].values.reshape(-1)
	### printing
	print('A:',np.shape(A))
	print('kibana_matrix:',np.shape(kibana_matrix))
	#print('y_train:',np.shape(y_train))
	#print('y_val:',np.shape(y_val))
	#print('y_test:',np.shape(y_test))

	np.savez(outfile, feats=feats, y_train = y_train, y_val = y_val, y_test = y_test, train_index = train_index, val_index = val_index, test_index = test_index)
	#print('lens y_train, y_val, y_test:',len(y_train), len(y_val), len(y_test))
	return pr_cols

#top = 50
#kibana_nodes, kibana_matrix = get_kibana_nodes_matrix(kibana_file,3)
#get_top_hashtags(kibana_matrix,top)

#pr_cols = get_features_adjM_data()


#kibana_matrix,A,feat_cols, pr_cols = get_ordered_features_and_adjM_matrices([-1,0,1], 'polarity_category', 'polarity_w_hashtags', 'hashtags', 20) 
#get_features_adjM_data(kibana_matrix,A,'polarity_w_hashtags',feat_cols, pr_cols)
#get_feat_prediction_cols('polarity_w_hashtags', 'hashtags', 'polarity_category', kibana_matrix)  
#customized_train_validate_test_split(5, df, train_percent=.6, validate_percent=.2, seed=None):
#kibana_nodes_matrix(kibana_file)
#nodes1 = get_networks_nodes(wiki_file)
#print(len(set(nodes1)))
#nodes2, kibana_matrix = get_kibana_nodes_matrix(kibana_file)
#print('Intersection:',len(set(nodes1).intersection(set(nodes2))))
#generate_adj_matrix(wiki_file)
#kibana_nodes, kibana_matrix = get_kibana_nodes_matrix(kibana_file)
[-1,0,1]
