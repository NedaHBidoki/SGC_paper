import pandas as pd
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
import numpy as np
import os


db = 'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/gcn/preprocess_DARPA_data/raw_data/%s/'%db
wiki_file = 'network_twitter_cve.csv'


def get_wiki_adj_matrix(wiki_file):
	df = pd.read_csv(base + wiki_file)
	nodes = df['node'].values.tolist()
	parents = df['parent'].values.tolist()
	all_nodes = set(nodes + parents)
	edges = zip(nodes,parents)
	G = nx.Graph()
	G.add_edges_from(list(edges))
	draw_net(G)
	#A = nx.adjacency_matrix(G)
	#A_pd = nx.to_pandas_adjacency(G, nodelist=all_nodes, dtype=int)
	#A_pd.to_csv(base + 'A_matrix.csv')
	#print('A_pd.shape', A_pd.shape)
	return G

def draw_net(G):
	nx.draw(G, pos=nx.spring_layout(G))	
	plt.savefig('plots/%s.pdf'%db)
	#plt.show()

G = get_wiki_adj_matrix(wiki_file)
draw_net(G)
