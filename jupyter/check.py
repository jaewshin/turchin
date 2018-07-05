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
import progressbar
import time 


pnas_data1 = pd.read_csv('pnas_data1.csv')
pnas_data2 = pd.read_csv('pnas_data2.csv')
raw = pd.read_csv('seshat-20180402.csv', encoding = 'ISO-8859-1')
raw_corrected = pd.read_csv('seshat_corrected.csv', encoding = 'ISO-8859-1')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']

filtered_pnas1 = StandardScaler().fit_transform(pnas_data1.loc[:, features].values)

def svd(data):
    """
    perform singular value decomposition on the given data matrix
    """
    #center the data
    mean = np.mean(data, axis=0)
    data -= mean
    
    P, D, Q = np.linalg.svd(filtered_pnas1, full_matrices=False)
#     col_vec = np.abs(Q)/sum(np.abs(Q))*100
#     first_col = col_vec[:, 0]

#     col_vec = np.abs(Q)/sum(np.abs(Q))*100
#     D = D/sum(D)*100  # percentage of singular values 
    
    return P, D, Q

P, D, Q = svd(filtered_pnas1)

product = np.dot(Q.T,Q)
np.fill_diagonal(product,0)
if (product.any() == 0): 
    raise Exception('not orthogonal') #check orthogonality of the matrix to ensure that PCs are orthogonal

D = [x**2/sum([y**2 for y in D]) for x in D] # variance for each vector 
data = np.matmul(filtered_pnas1, Q.T) # data matrix is obtained by multiplying initial data matrix with SVD column matrix


# retrive 25 polities from the lowest and the highest in the two Gaussian distrubitons 
# from the first principal component
subset_features = ['NGA', 'PolID', 'Time']

idx_data = sorted(range(len(data[:,0])), key = lambda i: data[:,0][i])
re_idx = pnas_data1.reindex(idx_data)
unique = re_idx.drop_duplicates(subset = subset_features) #drop duplicates from imputation

lowest = unique.head(25).loc[:,subset_features]
highest = unique.tail(25).loc[:,subset_features]
lowest.to_csv('lowest_25.csv')
highest.to_csv('highest_25.csv')

# retrive 25 polities from the left and right side of the transition axis (~-0.9)
lf_data = [idx for idx in unique.index.values if data[:,0][idx] <= -0.9][-25:]
rt_data = [idx for idx in unique.index.values if data[:,0][idx] > -0.9][:25]

lf = unique.loc[lf_data].loc[:, subset_features]
rt = unique.loc[rt_data].loc[:, subset_features]
lf.to_csv('left_transition.csv')
rt.to_csv('right_transition.csv')

#Gaussian Mixture Model 
#fit GMM
gmm = GMM(n_components=2)
gmm = gmm.fit(X=np.expand_dims(data[:,0], 1))

# Evaluate GMM
gmm_x = np.linspace(-6, 6, 200)
gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1,1)))

# mean and covariance for each component
gauss_one = gmm.weights_[0] #weight for gaussian distribution
gauss_two = gmm.weights_[1] #weight for gaussian distribution 

prob_distr = gmm.predict_proba(X=np.expand_dims(sorted(data[:,0]), 1))
ls = list()
# print(sorted(data[:,0])[:100])
i=0
print(gmm.weights_)
for idx in range(len(prob_distr)):


	if idx > 2860 and idx < 2900:
		print(prob_distr[idx])
		i+= 1

	if prob_distr[idx][0] <= prob_distr[idx][1]:
		ls.append(idx_data[idx])
print(i)
print(len(ls))