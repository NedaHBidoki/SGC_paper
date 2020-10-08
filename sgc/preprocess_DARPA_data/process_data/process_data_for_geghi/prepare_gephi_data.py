import pandas as pd
import sys
import random
import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os

db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/raw_data/%s/'%db
wiki_file = 'network_twitter_cve.csv'
kibana_file = 'attr_74074.csv'#'user_attr_matrix2.csv'

def create_node_list():
	df = pd.read_csv(base + '%s/network_%s.csv'%(db,db))
	print(df.columns)
	df = df.rename(columns={'node':'source','parent':'target'})
	print(df.columns)
	df[['source','target']].to_csv('data/nodes.csv', index = False)

def create_nodes_attrs_edges_files(kibana_file):
	df = pd.read_csv(base + kibana_file)#,nrows = 1000)
	df = df.fillna(0)
	df.columns = df.columns.str.replace(": Descending", "")
	df = df.rename(columns={'id_h':'label'})
	df['id'] = pd.Series(df['label']).astype('category').cat.codes.values
	df  = df.set_index('id')
	df.to_csv('data/nodet_attr.csv')
	df_nodes = pd.read_csv(base + wiki_file)
	df_nodes['node'] = pd.Series(df_nodes['node']).astype('category').cat.codes.values
	df_nodes['parent'] = pd.Series(df_nodes['parent']).astype('category').cat.codes.values
	df_nodes[['node','parent']].to_csv('data/edges.csv',index = False)
	
create_nodes_attrs_edges_files(kibana_file)
