import pandas as pd
import sys
import random
import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
import scipy.sparse
import numpy
import os


db = 'twitter_crypto'#'twitter_cve'
base = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/preprocess_DARPA_data/raw_data/%s/'%db
wiki_file = 'network_twitter_crypto.csv'


def get_G(wiki_file):
	df = pd.read_csv(base + wiki_file, nrows=2000)
	nodes = df['node'].values.tolist()
	parents = df['parent'].values.tolist()
	all_nodes = set(nodes + parents)
	edges = zip(nodes,parents)
	G = nx.Graph()
	G.add_edges_from(list(edges))
	return G

def G_stat(G):	
	comps = list(nx.connected_components(G))
	comps.sort(key=len, reverse= True)
	S = [G.subgraph(c).copy() for c in comps]
	
	# plot communties stat
	from networkx.algorithms.community import greedy_modularity_communities
	Communities = list(greedy_modularity_communities(S[0]))
	clens = [len(c) for c in Communities]
	print('communities len:', clens)
	_ = plt.hist(clens, bins='auto',color='gray')
	plt.ylabel('Size')
	plt.xlabel('Frequency')	
	plt.title("Community sizes in the largest component")
	plt.grid()
	plt.savefig(base+'comp_sizes.pdf')
	plt.show()

	
	largest_cc = max(comps, key=len)
	comp_sequence = sorted([len(c) for c in comps], reverse=True)
	# print "component sequence", component_sequence
	plt.loglog(comp_sequence, 'b-', marker='o')
	plt.title("Component rank plot")
	plt.ylabel("component size")
	plt.xlabel("rank")
	plt.grid()
	plt.savefig('component_rank_connected.pdf')
	plt.close()
	print(len(G),len(largest_cc))


	#print(nx.diameter(G)) #answer INF
	degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
	dmax = max(degree_sequence)
	print(dmax)
	plt.loglog(degree_sequence, 'b-', marker='o')
	plt.title("Degree rank plot")
	plt.ylabel("degree")
	plt.xlabel("rank")
	plt.grid()

	# draw graph in inset
	plt.axes([0.45, 0.45, 0.45, 0.45])
	Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
	pos = nx.spring_layout(Gcc)
	plt.axis('off')
	nx.draw_networkx_nodes(Gcc, pos, node_size=20)
	nx.draw_networkx_edges(Gcc, pos, alpha=0.4)

	plt.savefig(base + 'degree_rank_connected.pdf')
	plt.close()

	
	C = [G.subgraph(c).copy() for c in Communities]
	for i,c in enumerate(C):
		nx.draw(c)
		plt.savefig(base+ '%s.png'%i)

	sys.exit()
	for i,c in enumerate(C):
		nx.write_edgelist(c, base + "communities/community_%s_size%s.csv"%(i,len(c)), delimiter=',')
		df = pd.read_csv( base + "communities/community_%s_size%s.csv"%(i,len(c)), names=['node','parent','w'])
		df['created_at']='nan'
		df[['created_at','node','parent']].to_csv( base + "communities/community_%s_size%s.csv"%(i,len(c)),index=False)
	sys.exit()	


	# save components
	comps = list(nx.connected_components(G))
	comps.sort(key=len, reverse= True)
	S = [G.subgraph(c).copy() for c in comps]
	for i,g in enumerate(S[0:5]):
		nx.write_edgelist(G, "component_%s.csv"%i, delimiter=',')
		df = pd.read_csv("component_%s.csv"%i, names=['node','parent','w'])
		df['created_at']='nan'
		df[['created_at','node','parent']].to_csv("component_%s.csv"%i,index=False)


	# save components in a csv file
	comp_df = pd.DataFrame()
	comp_df['comp']= comps
	comp_df['comp #']=range(len(comps))
	comp_df['len'] = comp_df['comp'].apply(lambda x: len(x))
	comp_df.head().to_csv(base + '/connected_components.csv')
	print(comp_df['len'].head())
	

def random_walk(G, walkLength):
	pos=nx.spring_layout(G)
	A = nx.adj_matrix(G)
	A = A.todense()
	A = numpy.array(A, dtype = numpy.float64)
	# let's evaluate the degree matrix D
	D = numpy.diag(numpy.sum(A, axis=0))
	# ...and the transition matrix T
	T = numpy.dot(numpy.linalg.inv(D),A)
	# define the starting node, say the 0-th
	p = numpy.array([1]+ [0] * (len(G)-1)).reshape(-1,1)
	visited = list()
	for k in range(walkLength):
	    # evaluate the next state vector
	    p = numpy.dot(T,p)
	    # choose the node with higher probability as the visited node
	    visited.append(numpy.argmax(p))
	#nx.draw(G)  # networkx draw()
	print(visited)
	#nx.draw_networkx_nodes(G,pos,nodelist=visited,node_color='b',node_size=500,alpha=0.8)
	#plt.draw()
	#plt.savefig('network.png') 
	print(len(G), ' vs ', len(visited))
G= get_G(wiki_file)
G_stat(G)
#random_walk(G, 10)
