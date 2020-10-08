import pandas as pd
import sys
import random
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os
from collections import Counter


db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/raw_data/%s/'%db
wiki_file = 'network_twitter_cve.csv'
kibana_file = 'attr_74074.csv'#'user_attr_matrix2.csv'


def get_kibana_data_stat(kibana_file):
	df = pd.read_csv(base + kibana_file)#,nrows = 1000)
	df = df.fillna(0)
	df.columns = df.columns.str.replace(": Descending", "")
	df = df.rename(columns={'id_h':'label'})
	df['id'] = pd.Series(df['label']).astype('category').cat.codes.values
	df  = df.set_index('id')
	print(pd.Series(' '.join(df['hashtags.keyword']).lower().split()).value_counts()[:10])
	df_edge = pd.read_csv(base + wiki_file)
	df_edge['node'] = pd.Series(df_edge['node']).astype('category').cat.codes.values
	df_edge['parent'] = pd.Series(df_edge['parent']).astype('category').cat.codes.values

	


get_kibana_data_stat(kibana_file)
