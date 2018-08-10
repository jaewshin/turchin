import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM
from sklearn import linear_model
from sklearn.decomposition import PCA, FastICA
from multiprocessing import Pool
import os
# read csv/excel data files 
pnas_data1 = pd.read_csv("pnas_data1.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def svd(data):
    """
    perform singular value decomposition on the given data matrix
    """
    #center the data
    mean = np.mean(data, axis=0)
    data -= mean
    
    P, D, Q = np.linalg.svd(data, full_matrices=False)
    
    return P, D, Q

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

# flow vectors for each NGAs
def flow_vec(ngas, data, data_idx):
    """
    Find the average of the flow vectors 
    """
    nga_dict = {key:list() for key in ngas}
    vec_coef1 = list() # coefficients for overall flow vectors for each ngas in the first gaussian
    vec_ic1 = list() # intercept for overall flow vectors for each ngas in the first gaussian
    vec_coef2 = list() # coefficients for overall flow vectors for each ngas in the second gaussian
    vec_ic2 = list() # intercept for overall flow vectors for each ngas in the second gaussian

    for idx in range(len(data)):
        nga = pnas_data1.loc[idx].NGA
        nga_dict[nga].append((data[:,0][idx], data[:,1][idx], times[idx][0], data_idx[idx]))            

    for i in range(len(ngas)):

        nga_pc_gauss1 = [[p,j] for p,j,_,t in nga_dict[ngas[i]] if t in gauss1_idx]
        nga_pc_gauss2 = [[p,j] for p,j,_,t in nga_dict[ngas[i]] if t in gauss2_idx]

        nga_time1 = np.asarray([k for _,_,k,t in nga_dict[ngas[i]] if t in gauss1_idx])
        nga_time2 = np.asarray([k for _,_,k,t in nga_dict[ngas[i]] if t in gauss2_idx])
                 
        #fit linear regression vector
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
    
    # find the average coefficients and intercepts for each Gaussian
    gauss1_coef = np.mean(vec_coef1, axis=0)
    gauss1_ic = np.mean(vec_ic1, axis=0)
    gauss2_coef = np.mean(vec_coef2, axis=0)
    gauss2_ic = np.mean(vec_ic2, axis=0)

    gauss1_t = sorted(gauss1_time)
    gauss2_t = sorted(gauss2_time)
    ic1 = np.asarray([gauss1_ic]).T
    ic2 = np.asarray([gauss2_ic]).T

    def line_vec(time, coef, intercept):
        return [(coef*i+intercept) for i in time]
    
    gauss1 = line_vec(gauss1_t, gauss1_coef, ic1)
    gauss2 = line_vec(gauss2_t, gauss2_coef, ic2)

    gauss1_x= [i for i,j in gauss1 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]
    gauss1_y= [j for i,j in gauss1 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]
    gauss2_x= [i for i,j in gauss2 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]
    gauss2_y= [j for i,j in gauss2 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]

    ols1 = linear_model.LinearRegression()
    model1= ols1.fit(gauss1_x, gauss1_y)
    ols2 = linear_model.LinearRegression()
    model2= ols2.fit(gauss2_x, gauss2_y)

    return model1.coef_, model2.coef_

def pl(n):
    np.random.seed()
    
    data_idx = dict()
    
    resample = np.random.randint(0, len(data), size=len(data))
    resampled_mat = data[resample]

    data_idx = dict(enumerate(resample.tolist()))
    
    assert resampled_mat.shape == data.shape # check if the resampled matrix has same dimension as data matrix

    first_coef, sec_coef = flow_vec(ngas, resampled_mat, data_idx)
    first_ang, sec_ang = np.arctan(first_coef[0][0]), np.arctan(sec_coef[0][0])
    
    return first_ang, sec_ang

gauss1_idx, gauss2_idx = gaussian_idx(data_mat)
gauss1_time = [times[i] for i in gauss1_idx] # time for the first gaussian data
gauss2_time = [times[j] for j in gauss2_idx] # time for the second gaussian data
coef1, coef2 = flow_vec(ngas, data, {key:key for key in range(len(data))})
ang1, ang2 = np.arctan(coef1[0]), np.arctan(coef2[0])

def bstr_flow(ngas, data, n = 5000):
    """
    Bootstrap the angle between the flow vectors and the x axis for each cluster
    """
    p = Pool(os.cpu_count()-1)
    results = p.map(pl, range(n))
    angle1, angle2 = zip(*results)

    # angle1 = []; angle2 = []
    # for i in range(n):
    # 	a, b = pl()
    # 	angle1.append(a)
    # 	angle2.append(b)

    return angle1, angle2

angle1, angle2 = bstr_flow(ngas, data)

with open('angle1.txt', 'w') as f:  
    for angle in angle1:
        f.write('%s\n' % angle)

with open('angle2.txt', 'w') as f:
	for angle in angle2:
		f.write('%s\n' % angle)

