from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh,ArpackNoConvergence

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(path='data/cora',dataset='cora'):
    print('loading {} dataset'.format(dataset)
