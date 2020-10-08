import scipy.sparse
import numpy as np
import random
import sys


# generate a list of numbers
def int_gen():
	Start = 1
	Stop = 10
	limit = 1000
	RandomI_ListOfIntegers = [random.randrange(Start, Stop) for iter in range(limit)]
	return RandomI_ListOfIntegers


# generate a list of numbers
def bin_gen():
	Start = 1
	Stop = 2
	limit = 1000
	RandomI_ListOfIntegers = [random.randrange(Start, Stop) for iter in range(limit)]
	return RandomI_ListOfIntegers

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return perm[:train_end], perm[train_end:validate_end], perm[validate_end:]

# generate a sysnthetic adj matrix
sparse_matrix = scipy.sparse.csc_matrix(np.array([bin_gen() for i in range(0,1000)]))
print(sparse_matrix.shape)
sparse_matrix.todense()
scipy.sparse.save_npz('reddit_adj.npz', sparse_matrix)
sparse_matrix = scipy.sparse.load_npz('reddit_adj.npz')


# generate a systhetic features dataframe
import pandas as pd
df = {'col_%s'%i: int_gen() for i in range(1,10)}
df = pd.DataFrame(df)



# generate a systhetic feature matrix
from tempfile import TemporaryFile
outfile = 'reddit.npz' #TemporaryFile()
feats = df[['col_%s'%i for  i in range(1,5)]].values
y = df[['col_%s'%i for  i in range(5,6)]]
train_index, val_index, test_index = train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None)
y_train, y_val, y_test = y.iloc[train_index].values.reshape(-1), y.iloc[val_index].values.reshape(-1), y.iloc[test_index].values.reshape(-1)
print(y_train.shape)
print(train_index.shape)
sys.exit()
np.savez(outfile, feats=feats, y_train = y_train, y_val = y_val, y_test = y_test, train_index = train_index, val_index = val_index, test_index = test_index)
#_ = outfile.seek(0) # Only needed here to simulate closing & reopening file
npzfile = np.load(outfile)
print(npzfile.files)

print(npzfile['feats'])
