import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.stats import kstest, anderson
import math
import seaborn as sns; sns.set()
import progressbar as pb
import time 
from scipy.sparse import csgraph

# read csv/excel data files 
pnas_data1 = pd.read_csv('/home/jaeweon/research/data/pnas_data1.csv')
pnas_data2 = pd.read_csv('/home/jaeweon/research/data/pnas_data2.csv')
raw = pd.read_excel('/home/jaeweon/research/data/raw.xlsx', encoding = 'ISO-8859-1')
raw_corrected = pd.read_csv('/home/jaeweon/research/data/seshat_corrected.csv', encoding = 'ISO-8859-1')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# format data 

# extract 9 Complexity Characteristic variables 
features = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']

# take subset of original data table with 9 CCs and change it into numpy array 
data_mat = pnas_data1.loc[:, features].values
scaled = StandardScaler().fit_transform(data_mat)

# class for progressbar
class progress_timer:

    def __init__(self, n_iter, description="Something"):
        self.n_iter = n_iter
        self.iter = 0
        self.description = description + ': '
        self.timer = None
        self.initialize()

    def initialize(self):
        #initialize timer
        widgets = [self.description, pb.Percentage(), ' ',   
                   pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
        self.timer = pb.ProgressBar(widgets=widgets, maxval=self.n_iter).start()

    def update(self, q=1):
        #update timer
        self.timer.update(self.iter)
        self.iter += q

    def finish(self):
        #end timer
        self.timer.finish()

def svd(data):
    """
    perform singular value decomposition on the given data matrix
    """
    #center the data
    mean = np.mean(data, axis=0)
    data -= mean
    
    P, D, Q = np.linalg.svd(data, full_matrices=False)
    
    return P, D, Q

# SVD on the original data matrix. 
filtered_pnas1 = StandardScaler().fit_transform(pnas_data1.loc[:, features].values)

P, D, Q = svd(filtered_pnas1)
reconstruct = np.matmul(np.matmul(P, np.diag(D)), Q)
print(np.std(filtered_pnas1), np.std(reconstruct), np.std(filtered_pnas1-reconstruct)) # check if the reconstructed matrix is valid 



