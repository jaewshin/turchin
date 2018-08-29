import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM
from sklearn import linear_model
import time 
import math
from sklearn.decomposition import PCA, FastICA
import os
from multiprocessing import Pool, Process

# read csv/excel data files 
pnas_data1 = pd.read_csv("pnas_data1.csv")

# format data 
# extract 9 Complexity Characteristic variables 
features = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']

# take subset of original data table with 9 CCs and change it into numpy array 
data_mat = StandardScaler().fit_transform(pnas_data1.loc[:, features].values)
times = pnas_data1.loc[:, ['Time']].values
ngas = pnas_data1.NGA.unique().tolist()
P, D, Q = svd(data_mat)
data = np.matmul(data_mat, Q.T)

def gaussian_idx(data_mat):
	"""
	Return index for data points that lie in one of the two Gaussians
	"""
	#Gaussian Mixture Model 
	#fit GMM
	gmm = GMM(n_components=2).fit(data_mat)
	cov = gmm.covariances_
	prob_distr = gmm.predict_proba(data_mat)

	# determine to which of the two gaussians each data point belongs by looking at probability distribution 
	if gmm.weights_[0] < gmm.weights_[1]:
		gauss1_idx = [i for i in range(len(prob_distr)) if prob_distr[i][0] >= prob_distr[i][1]]
		gauss2_idx = [j for j in range(len(prob_distr)) if prob_distr[j][1] >= prob_distr[j][0]]
	else:
		gauss1_idx = [i for i in range(len(prob_distr)) if prob_distr[i][0] <= prob_distr[i][1]]
		gauss2_idx = [j for j in range(len(prob_distr)) if prob_distr[j][1] <= prob_distr[j][0]]
	return gauss1_idx, gauss2_idx

def avg_flowvec(data, gauss1_idx, gauss2_idx):
	"""
	Compute the slope and intercept for average flow vectors for each cluster
	"""
	# flow vectors for each NGAs
	nga_dict = {key:list() for key in ngas}
	vec_coef1 = list() # coefficients for overall flow vectors for each ngas in the first gaussian
	vec_ic1 = list() # intercept for overall flow vectors for each ngas in the first gaussian
	vec_coef2 = list() # coefficients for overall flow vectors for each ngas in the second gaussian
	vec_ic2 = list() # intercept for overall flow vectors for each ngas in the second gaussian

	for idx in range(len(data)):
		nga = pnas_data1.loc[idx].NGA
		nga_dict[nga].append((data[:,0][idx], data[:,1][idx], times[idx][0], idx))            

	for i in range(len(ngas)):  # flow vector for each NGA

		nga_pc1 = [p for p,_,_,_ in nga_dict[ngas[i]]] 
		nga_pc2 = [j for _,j,_,_ in nga_dict[ngas[i]]]
		nga_time = [k for _,_,k,_ in nga_dict[ngas[i]]]

		nga_pc1 = [x for _,x,_ in sorted(zip(nga_time, nga_pc1, nga_pc2))]
		nga_pc2 = [y for _,_,y in sorted(zip(nga_time, nga_pc1, nga_pc2))]

		nga_pc_gauss1 = [[p,j] for p,j,_,t in nga_dict[ngas[i]] if t in gauss1_idx]
		nga_pc_gauss2 = [[p,j] for p,j,_,t in nga_dict[ngas[i]] if t in gauss2_idx]

		nga_time1 = np.asarray([k for _,_,k,t in nga_dict[ngas[i]] if t in gauss1_idx])
		nga_time2 = np.asarray([k for _,_,k,t in nga_dict[ngas[i]] if t in gauss2_idx])

		xfit1 = np.linspace(-6, 4, len(nga_time1))
		xfit2 = np.linspace(-6, 4, len(nga_time2))

		assert len(nga_pc_gauss1) == len(nga_time1)
		assert len(nga_pc_gauss2) == len(nga_time2)

		#fit linear regression 
		if len(nga_time1) == 0:
			ols2 = linear_model.LinearRegression()
			model2 = ols2.fit(nga_time2.reshape(-1,1), nga_pc_gauss2)

			vec_coef2.append(model2.coef_)
			vec_ic2.append(model2.intercept_)

		elif len(nga_time2) == 0:
			ols1 = linear_model.LinearRegression()
			model1 = ols1.fit(nga_time1.reshape(-1,1), nga_pc_gauss1)

			vec_coef1.append(model1.coef_)
			vec_ic1.append(model1.intercept_)

		else:
			ols1 = linear_model.LinearRegression()
			model1 = ols1.fit(nga_time1.reshape(-1,1), nga_pc_gauss1)
			ols2 = linear_model.LinearRegression()
			model2 = ols2.fit(nga_time2.reshape(-1,1), nga_pc_gauss2)

			vec_coef1.append(model1.coef_)
			vec_ic1.append(model1.intercept_)
			vec_coef2.append(model2.coef_)
			vec_ic2.append(model2.intercept_)

	gauss1_coef = np.mean(vec_coef1, axis=0)
	gauss1_ic = np.mean(vec_ic1, axis=0)
	gauss2_coef = np.mean(vec_coef2, axis=0)
	gauss2_ic = np.mean(vec_ic2, axis=0)

	# slope and intercept of the average flow vector for each cluster
	slope1 = gauss1_coef[1]/gauss1_coef[0]
	slope2 = gauss2_coef[1]/gauss2_coef[0]
	intercept1 = gauss1_ic[1] - ((slope1) * gauss1_ic[0])
	intercept2 = gauss2_ic[1] - ((slope2) * gauss2_ic[0])

	return slope1, intercept1, slope2, intercept2 

def avg_flow_bstr(data, n=5000):
    """
    Bootstrap the average flow vectors 
    """
    slope1_ag = []; slope2_ag = []
    intercept1_ag = []; intercept2_ag = []
    
    gauss1_idx, gauss2_idx = gaussian_idx(data_mat)

    for i in range(n):
        
        #resampled matrix
        resample = np.random.randint(0, len(data), size = len(data))
        resampled_mat = data[resample]
        
        # find the average flow vector for each cluster
        slope1, intercept1, slope2, intercept2 = avg_flowvec(resampled_mat, gauss1_idx, gauss2_idx)
        
        slope1_ag.append(slope1); slope2_ag.append(slope2)
        intercept1_ag.append(intercept1); intercept2_ag.append(intercept2)
                
    return slope1_ag, slope2_ag, intercept1_ag, intercept2_ag

slope1_ag, slope2_ag, intercept1_ag, intercept2_ag = avg_flow_bstr(data)

with open('slope1.txt', 'w') as f:  
    for slope in slope1_ag:
        f.write('%s\n' % slope)

with open('slope2.txt', 'w') as f:
    for slope in slope2_ag:
        f.write('%s\n' % slope)

with open('intercept1.txt', 'w') as f:  
    for ic1 in intercept1_ag:
        f.write('%s\n' % ic1)

with open('intercept2.txt', 'w') as f:
    for ic2 in intercept2_ag:
        f.write('%s\n' % ic2)

