import sys
benchmark_dirs = '/home/social-sim/Documents/SocialSimCodeTesting/GCN/sgc/banchmarks'
sys.path.insert(0, benchmark_dirs)
from clsf_benchmarks import *


data = initialization(benchmark_dirs + '/iris.data')
X,y = get_X_y(data)
X_train, X_test, y_train, y_test = split_data(X,y)
evaluate_benchmarks(X_train, X_test, y_train, y_test, 'test2', 'test2', 'test2','test_plots')


