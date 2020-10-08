import pandas as pd
import sys 
import random
import networkx as nx
from operator import itemgetter
import matplotlib.pyplot as plt
db = 'twitter'
base = '/home/social-sim/Documents/SocialSimCodeTesting/TH/TH-analysis/codes/GCNN/data/%s/'%db
domains=['cve','crypto','cyber']
df = pd.DataFrame()

for d in domains:
	data = pd.read_csv(base + 'network_twitter_%s.csv'%d)
	df = df.append(data, ignore_index=True)
	print(len(df))

df.to_csv(base + '3domains_networks.csv',index= False)
