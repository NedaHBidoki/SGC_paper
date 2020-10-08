import pandas as pd
import sys
import random
import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os
import csv

print ("the prdicted column index %s" % (sys.argv[1]))
db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/raw_data/%s/'%db
wiki_file = 'network_twitter_cve.csv'
kibana_file = 'attr_74074.csv'#'user_attr_matrix2.csv'


def get_networks_nodes(wiki_file):
	df = pd.read_csv(base + wiki_file)
	nodes = df['node'].values.tolist()
	parents = df['parent'].values.tolist()
	nodes.extend(parents)
	return nodes


def get_kibana_nodes_matrix(kibana_file):
	#df = pd.read_csv(base + kibana_file)
	rm_quote = lambda x: x.replace('"', '')
	df = pd.read_csv(base + kibana_file,nrows = 1000)
	df = df.fillna(0)
	df.columns = df.columns.str.replace(": Descending", "")
	df  = df.set_index('id_h')
	print(df.dtypes)
	column_names_for_onehot = ["user.time_zone.keyword", "hashtags.keyword","possibly_sensitive"]
	dum_df  = pd.get_dummies(df,columns=column_names_for_onehot, dummy_na=True)
	dum_df = dum_df.groupby(dum_df.index).max()
	#print([c for c in dum_df.columns if 'sensiti' in c])
	#print(dum_df['possibly_sensitive_True'].head())
	nodes = df.index
	return nodes, dum_df

def get_wiki_adj_matrix(wiki_file):
	df = pd.read_csv(base + wiki_file)
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

def intersection_data(A_pd,kibana_matrix):
	A_pd = A_pd[A_pd.index.isin(kibana_matrix.index)]
	A_pd = A_pd[np.intersect1d(A_pd.columns, kibana_matrix.index)]
	kibana_matrix = kibana_matrix[kibana_matrix.index.isin(A_pd.index)]
	print('A_pd.shape', A_pd.shape)
	print('kibana_matrix.shape', kibana_matrix.shape)
	return A_pd, kibana_matrix

def get_ordered_adj_and_attr_matrices():
	A_pd = get_wiki_adj_matrix(wiki_file)
	kibana_nodes, kibana_matrix = get_kibana_nodes_matrix(kibana_file)
	A_pd, kibana_matrix = intersection_data(A_pd,kibana_matrix)
	kibana_matrix = kibana_matrix.reindex(A_pd.index)
	#print('A_pd.index:', (A_pd).head())
	#print('kibana_matrix.index:', (kibana_matrix).head())
	print('A_pd.index:', A_pd.index[:5])
	print('kibana_matrix.index:', kibana_matrix.index[:5])
	print('%%%%%%%%%%%%%%%%%%%%%%%%%')
	print(A_pd.index.equals(kibana_matrix.index))
	print(A_pd.index.difference(kibana_matrix.index))
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
	return kibana_matrix,A_pd.values

def save_npz_adj_matrix(M, name, dir_):
	sparse_matrix = scipy.sparse.csc_matrix(M)
	print(sparse_matrix.shape)
	sparse_matrix.todense()
	scipy.sparse.save_npz(dir_ + '%s_adj.npz'%name, sparse_matrix)
	sparse_matrix = scipy.sparse.load_npz(dir_ + '%s_adj.npz'%name)

def draw_net(G):
	nx.draw(G, pos=nx.spring_layout(G))
	plt.show()

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    print('train_inx:',perm[:train_end])
    print('val_inx:',perm[train_end:validate_end])
    print('test_inx:',perm[validate_end:])
    return perm[:train_end], perm[train_end:validate_end], perm[validate_end:]

def customized_train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    #colname = df.columns[pr_col] 
    colname = sys.argv[1]
    print('colname:',colname)
    one_value_list = df.index[df[colname] == 1].tolist()
    if len(one_value_list)<3:
        sys.exit()
    one_tr,one_va, one_te =  np.array_split(np.array(one_value_list),3)
    #one_tr,one_va, one_te = list(one_tr),list(one_va), list(one_te)
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    perm = np.setdiff1d(perm,np.array(one_value_list))
    m = len(perm)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    #train = df.iloc[perm[:train_end]]
    #validate = df.iloc[perm[train_end:validate_end]]
    #test = df.iloc[perm[validate_end:]]
    print(type(perm[:train_end]))
    train_inx = np.concatenate((perm[:train_end], one_tr), axis=None) 
    val_inx = np.concatenate((perm[train_end:validate_end], one_va), axis=None)  
    test_inx = np.concatenate((perm[validate_end:], one_te), axis=None) 
    print('one_tr:',one_tr)
    print('one_va:',one_va)
    print('one_te:',one_te)  
    print('train_inx:',train_inx)
    print('val_inx:',val_inx)
    print('test_inx:',test_inx)

    print('one_value_list:',len(one_value_list))  
    print('perm:',len(perm))  
    print('one_tr:',len(one_tr))
    print('one_va:',len(one_va))
    print('one_te:',len(one_te)) 
    print('train_inx:',len(train_inx))
    print('val_inx:',len(val_inx))
    print('test_inx:',len(test_inx))
    return train_inx, val_inx, test_inx

def get_top_hashtags(kibana_matrix):
	cols = list(kibana_matrix.columns)
	keyword = 'hashtags'
	feat_cols = [c for c in cols if (keyword in c) and ('nan' not in c)]
	print('m largest:',type(kibana_matrix[feat_cols].sum().nlargest(13, keep='first')))
	top_hashtags = kibana_matrix[feat_cols].sum().nlargest(3, keep='first')
	top_hashtags.to_csv('top_hashtags/n_largest_hashtags.csv')
	top_hashtags = top_hashtags.index.tolist()
	#top_hashtags_indexes = [feat_cols.index(c) for c in top_hashtags if c in feat_cols]
	#print('top_hashtags_indexes:',top_hashtags)
	#resultFile = open('top_hashtags/n_largest_hashtags_indexes.csv','w')
	#for r in top_hashtags_indexes:
	#    resultFile.write(r + "\n")
	#resultFile.close()
	return top_hashtags

def get_feat_prediction_cols(kibana_matrix):
	cols = list(kibana_matrix.columns)
	keyword = 'hashtags'
	feat_cols = [c for c in cols if (keyword in c) and ('nan' not in c)]
	if sys.argv[1] == 'polarity':
		pr_cols = 'extension.polarity'
	elif sys.argv[1] == 'sensitivity':
		pr_cols= 'possibly_sensitive_True'
	#elif pr_col_num > len(feat_cols):
	#	print('index out of bound')
	#	sys.exit()
	else:
		#pr_cols_number = int(sys.argv[1]) #random.randrange(len(feat_cols))
		#pr_cols = feat_cols[pr_cols_number]
		pr_cols = sys.argv[1]
		feat_cols.remove(pr_cols)
		with open("predicted hashtags.csv", "a") as myfile:
			myfile.write("%s\t%s\n"%(pr_cols_number, pr_cols))
	print('number of predictors (hashtags): %s'%len(feat_cols))
	print('!!!!!!!!!!!!!!!! variable',pr_cols)
	return feat_cols, pr_cols


def get_features_adjM_data():
	kibana_matrix,A = get_ordered_adj_and_attr_matrices()
	print(len(kibana_matrix))
	print(kibana_matrix.sum())
	dir_ = 'npz_data/%s/'%db
	if not os.path.exists(dir_):
		os.makedirs(dir_)
	outfile = dir_ + '%s.npz'%db
	save_npz_adj_matrix(A, db, dir_)
	feat_cols, pr_cols = get_feat_prediction_cols(kibana_matrix)
	feats = kibana_matrix[feat_cols].values
	y = kibana_matrix[pr_cols]
	train_index, val_index, test_index = train_validate_test_split(kibana_matrix, train_percent=.6, validate_percent=.2, seed=None)
	y_train, y_val, y_test = y.iloc[train_index].values.reshape(-1), y.iloc[val_index].values.reshape(-1), y.iloc[test_index].values.reshape(-1)
	np.savez(outfile, feats=feats, y_train = y_train, y_val = y_val, y_test = y_test, train_index = train_index, val_index = val_index, test_index = test_index)
	print('lens y_train, y_val, y_test:',len(y_train), len(y_val), len(y_test))
	return pr_cols

#pr_cols = get_kibana_adj_data()

kibana_matrix,A = get_ordered_adj_and_attr_matrices()
top_hashtags = get_top_hashtags(kibana_matrix)
print('top_hashtags[0]:',top_hashtags[0])
sys.argv[1]= top_hashtags[0]
print('sys.argv[1]:',sys.argv[1])
kibana_matrix,A = get_ordered_adj_and_attr_matrices()
train_validate_test_split(kibana_matrix, train_percent=.6, validate_percent=.2, seed=None)


#kibana_nodes_matrix(kibana_file)
#nodes1 = get_networks_nodes(wiki_file)
#print(len(set(nodes1)))
#nodes2, kibana_matrix = get_kibana_nodes_matrix(kibana_file)
#print('Intersection:',len(set(nodes1).intersection(set(nodes2))))
#generate_adj_matrix(wiki_file)
#kibana_nodes, kibana_matrix = get_kibana_nodes_matrix(kibana_file)

