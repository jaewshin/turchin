import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
import matplotlib
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
import os
import math
import turchin
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA, FastICA
from sklearn import linear_model

# read csv/excel data files 
CC_file = os.path.abspath(os.path.join("./..","data","pnas_data1.csv")) #20 imputed sets
PC1_file = os.path.abspath(os.path.join("./..","data","pnas_data2.csv")) #Turchin's PC1s

CC_df = pd.read_csv(CC_file) # A pandas dataframe
PC1_df = pd.read_csv(PC1_file) # A pandas dataframe

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Subset only the 9 CCs and convert to a numpy array 
CC_names = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']
CC_array = CC_df.loc[:, CC_names].values

# Normalize the data (across 20 imputations, not within each imputation)
CC_scaled = StandardScaler().fit_transform(CC_array)
CC_times = CC_df.loc[:, ['Time']].values

# Do a singular value decomposition
P, D, Q = turchin.svd(CC_scaled)

# For each polity, project onto the principle components
# PC_matrix is 8280 x 9 = (414*20) x 9
PC_matrix = np.matmul(CC_scaled, Q.T)

NGAs = CC_df.NGA.unique().tolist() # list of unique NGAs from the dataset

# histogram after projecting 9-d vectors into the main principal component, for each of the 20 imputed sets
numImpute = 20
impute = ['V'+str(i) for i in range(1, numImpute+1)]

# For illustrative plots, show only 
# Random number seed from random.org between 1 and 1,000,000
np.random.seed(780745)
i = np.random.randint(numImpute)


# histogram of PC1 for pooled imputations
num_bins = 50
n, bins, patches = plt.hist(PC_matrix[:,0], num_bins, density=1, facecolor='blue', alpha=0.5)
plt.title("Pooled over imputations")
plt.xlabel("Projection onto first Principle Component")
plt.legend()
fileStem = "pc1_histogram_impute_pooled"
plt.savefig(fileStem + ".png")
plt.savefig(fileStem + ".eps")
plt.close()

# Create a dictionary that maps from World Region to Late, Intermediate, and Early NGAs
regionDict = {"Africa":["Ghanaian Coast","Niger Inland Delta","Upper Egypt"]}
regionDict["Europe"] = ["Iceland","Paris Basin","Latium"]
regionDict["Central Eurasia"] = ["Lena River Valley","Orkhon Valley","Sogdiana"]
regionDict["Southwest Asia"] = ["Yemeni Coastal Plain","Konya Plain","Susiana"]
regionDict["South Asia"] = ["Garo Hills","Deccan","Kachi Plain"]
regionDict["Southeast Asia"] = ["Kapuasi Basin","Central Java","Cambodian Basin"]
regionDict["East Asia"] = ["Southern China Hills","Kansai","Middle Yellow River Valley"]
regionDict["North America"] = ["Finger Lakes","Cahokia","Valley of Oaxaca"]
regionDict["South America"] = ["Lowland Andes","North Colombia","Cuzco"]
regionDict["Oceania-Australia"] = ["Oro PNG","Chuuk Islands","Big Island Hawaii"]

worldRegions = list(regionDict.keys()) # List of world regions

xmin = -10000
xmax = 2000
ymin = -7
ymax = 7
# Create a 5 x 2 plot to show time sequences organized by world region
f, axes = plt.subplots(int(len(worldRegions)/2),2, sharex=True, sharey=True,figsize=(12,15))
axes[0,0].set_xlim([xmin,xmax])
axes[0,0].set_ylim([ymin,ymax])
for i,reg in enumerate(worldRegions):
    regList = list(reversed(regionDict[reg]))
    # m,n index the subplots
    m = i % int(len(worldRegions)/2) # mod; result is 0, 1, 2, 3, or 4
    n = i // int(len(worldRegions)/2) # integer division; result is 0 or 1
    for nga in regList:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        pc1 = list()
        for t in times:
            ind = indNga & (CC_df['Time']==t) # boolean vector for slicing also by time
            pc1.append(np.mean(PC_matrix[ind,0]))
            
        axes[m,n].scatter(times,pc1,s=10)
    s = '{' + regList[0] + '; ' + regList[1] + '; ' + regList[2] + '}'
    axes[m,n].set_title(s,fontsize=10)
    if m != 4:
        plt.setp(axes[m,n].get_xticklabels(), visible=False)
    else:
        axes[m,n].set_xlabel("Calendar Date [AD]")
    if n == 0:
        axes[m,n].set_ylabel("PC1")
f.subplots_adjust(hspace=.5)
plt.savefig("pc1_vs_time_stacked_by_region.pdf")
plt.savefig("pc1_vs_time_stacked_by_region.eps")
plt.savefig("pc1_vs_time_stacked_by_region.png")
plt.close()

early = [regionDict[reg][2] for reg in worldRegions]
middle = [regionDict[reg][1] for reg in worldRegions]
late = [regionDict[reg][0] for reg in worldRegions]

allStarts = (early,middle,late)
startString = ['Early','Middle','Late']

# Create another plot of time sequences organized by early/intermediate/late onset of political centralization
f, axes = plt.subplots(len(allStarts),1, sharex=True, sharey=True,figsize=(4,10))
axes[0].set_xlim([xmin,xmax])
axes[0].set_ylim([ymin,ymax])
for i,start in enumerate(allStarts):
    for nga in start:
        indNga = CC_df["NGA"] == nga # boolean vector for slicing by NGA
        times = sorted(np.unique(CC_df.loc[indNga,'Time'])) # Vector of unique times
        pc1 = list()
        for t in times:
            ind = indNga & (CC_df['Time']==t) # boolean vector for slicing also by time
            pc1.append(np.mean(PC_matrix[ind,0]))
        axes[i].scatter(times,pc1,s=10)
    axes[i].set_title(startString[i],fontsize=10)
plt.savefig("pc1_vs_time_stacked_by_start.pdf")
plt.savefig("pc1_vs_time_stacked_by_start.eps")
plt.savefig("pc1_vs_time_stacked_by_start.png")
plt.close()

#Gaussian Mixture Model 
#fit GMM
gmm = GMM(n_components=2).fit(CC_scaled)
cov = gmm.covariances_
prob_distr = gmm.predict_proba(CC_scaled)

# determine to which of the two gaussians each data point belongs by looking at probability distribution 
if gmm.weights_[0] < gmm.weights_[1]:
    gauss1_idx = [i for i in range(len(prob_distr)) if prob_distr[i][0] >= prob_distr[i][1]]
    gauss2_idx = [j for j in range(len(prob_distr)) if prob_distr[j][1] >= prob_distr[j][0]]
else:
    gauss1_idx = [i for i in range(len(prob_distr)) if prob_distr[i][0] <= prob_distr[i][1]]
    gauss2_idx = [j for j in range(len(prob_distr)) if prob_distr[j][1] <= prob_distr[j][0]]


gauss1_time = [CC_times[i] for i in gauss1_idx] # time for the first gaussian data
gauss2_time = [CC_times[j] for j in gauss2_idx] # time for the second gaussian data

gauss1_point = [CC_scaled[i] for i in gauss1_idx] # 9-d data point for the first gaussian
gauss2_point = [CC_scaled[j] for j in gauss2_idx] # 9-d data point for the second gaussian

def dummy(data, ngas):
    """
    Given a gaussian projection data and a list of unique ngas, 
    """
    # dummy variables for NGAs for fixed effects
    dummy = [[1 if CC_df.loc[[point]].NGA.tolist()[0] == nga else 0 for nga in ngas] for point in data]
    return np.asarray(dummy)

dummy1 = dummy(gauss1_idx, NGAs)
dummy2 = dummy(gauss2_idx, NGAs)

gmm = GMM(n_components=2).fit(PC_matrix)
cov = gmm.covariances_

# main eigenvectors for covariances of each gaussians
eigval1, eigvec1 = np.linalg.eig(cov[0])
eigval2, eigvec2 = np.linalg.eig(cov[1])

# find the eigenvector corresponding to the largest eigenvalue for each of the two gaussians
max_eigvec1 = eigvec1[:, np.argmax(max(eigval1))] 
max_eigvec2 = eigvec2[:, np.argmax(max(eigval2))]

# max_eigvec1 = np.asarray([(i**2)/math.sqrt(sum([k**2 for k in max_eigvec1])) for i in max_eigvec1])
# max_eigvec2 = np.asarray([(j**2)/math.sqrt(sum([k**2 for k in max_eigvec2])) for j in max_eigvec2])

gauss1_proj = np.matmul(gauss1_point, max_eigvec1)
gauss2_proj = np.matmul(gauss2_point, max_eigvec2)

gauss1_proj = np.vstack((gauss1_proj.T, dummy1.T)).T
gauss2_proj = np.vstack((gauss2_proj.T, dummy2.T)).T

assert gauss1_proj.shape[1] == 31
assert gauss2_proj.shape[1] == 31

# Multiple linear regression over time
ols1 = linear_model.LinearRegression()
ols2 = linear_model.LinearRegression()
model1 = ols1.fit(gauss1_time, gauss1_point)
model2 = ols2.fit(gauss2_time, gauss2_point)

print("coefficients for the first gaussian: ", model1.coef_)
print("intercept for the first gaussian: ", model1.intercept_)
print("coefficients for the second gaussian: ",  model2.coef_)
print("intercept for the second gaussian: ", model2.intercept_)

# pca components 
pca = PCA(n_components=9)
pca.fit(CC_scaled)
components = pca.components_
print(components)
print(pca.components_.T * np.sqrt(pca.explained_variance_))

# calculate angle between two vectors 
def angle(vec1, vec2):
    """
    Given two vectors, compute the angle between the vectors
    """
    assert vec1.shape == vec2.shape
    
    cos_vec = np.inner(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    angle = math.acos(cos_vec)
    in_deg = math.degrees(angle)
    if in_deg >= 90:
        return (180-in_deg)
    return in_deg

# angle between the two main eigenvectors 
# max_eigvec1 = np.asarray([(i**2)/sum([k**2 for k in max_eigvec1]) for i in max_eigvec1])
# max_eigvec2 = np.asarray([(i**2)/sum([k**2 for k in max_eigvec2]) for i in max_eigvec2])

print("original", angle(max_eigvec1, max_eigvec2))
angles = []
for i in range(len(max_eigvec1)):
    norm1 = [max_eigvec1[j] if j != i else 0.0 for j in range(len(max_eigvec1))]
    norm2 = [max_eigvec2[j] if j != i else 0.0 for j in range(len(max_eigvec2))]
    print("angle after dropping %s th component: " %(i+1), angle(np.asarray(norm1), np.asarray(norm2)))
    angles.append(angle(np.asarray(norm1), np.asarray(norm2)))



# examine where the angle between the two main eigenvectors for each gaussian comes from
comp1 = np.matmul(max_eigvec1.T, components)
norm_comp1 = np.asarray([(i**2)/sum([k**2 for k in comp1]) for i in comp1])
comp2 = np.matmul(max_eigvec2.T, components)
norm_comp2 = np.asarray([(j**2)/sum([k**2 for k in comp2]) for j in comp2])

print("main eigenvector for the first Gaussian: \n", norm_comp1)
print("main eigenvector for the second Gaussian: \n",norm_comp2)

angles = []
print("original angle", angle(norm_comp1, norm_comp2))
for i in range(len(norm_comp1)): # angle using only some components
    norm1 = [norm_comp1[j] if j != i else 0.0 for j in range(len(norm_comp1))]
    norm2 = [norm_comp2[j] if j != i else 0.0 for j in range(len(norm_comp2))]
    print("angle after dropping %s th component: " %(i+1), angle(np.asarray(norm1), np.asarray(norm2)))
    angles.append(angle(np.asarray(norm1), np.asarray(norm2)))
    
#plt.plot(range(1,10), angles)
#plt.show()
#plt.close()

# flow vectors for each NGAs
nga_dict = {key:list() for key in NGAs}
vec_coef1 = list() # coefficients for overall flow vectors for each ngas in the first gaussian
vec_ic1 = list() # intercept for overall flow vectors for each ngas in the first gaussian
vec_coef2 = list() # coefficients for overall flow vectors for each ngas in the second gaussian
vec_ic2 = list() # intercept for overall flow vectors for each ngas in the second gaussian

for idx in range(len(PC_matrix)):
    nga = CC_df.loc[idx].NGA
    nga_dict[nga].append((PC_matrix[:,0][idx], PC_matrix[:,1][idx], CC_times[idx][0], idx))            

for i in range(len(NGAs)):  # flow vector for each NGA

    nga_pc1 = [p for p,_,_,_ in nga_dict[NGAs[i]]] 
    nga_pc2 = [j for _,j,_,_ in nga_dict[NGAs[i]]]
    nga_time = [k for _,_,k,_ in nga_dict[NGAs[i]]]

    nga_pc1 = [x for _,x,_ in sorted(zip(nga_time, nga_pc1, nga_pc2))]
    nga_pc2 = [y for _,_,y in sorted(zip(nga_time, nga_pc1, nga_pc2))]

    nga_pc_gauss1 = [[p,j] for p,j,_,t in nga_dict[NGAs[i]] if t in gauss1_idx]
    nga_pc_gauss2 = [[p,j] for p,j,_,t in nga_dict[NGAs[i]] if t in gauss2_idx]

    nga_time1 = np.asarray([k for _,_,k,t in nga_dict[NGAs[i]] if t in gauss1_idx])
    nga_time2 = np.asarray([k for _,_,k,t in nga_dict[NGAs[i]] if t in gauss2_idx])

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
        
        yfit2 = model2.predict(np.sort(nga_time2).reshape(-1, 1))
        #plt.plot([p for p, _ in yfit2], [q for _, q in yfit2])
        
        vec_coef2.append(model2.coef_)
        vec_ic2.append(model2.intercept_)
        
    elif len(nga_time2) == 0:
        ols1 = linear_model.LinearRegression()
        model1 = ols1.fit(nga_time1.reshape(-1,1), nga_pc_gauss1)
        
        vec_coef1.append(model1.coef_)
        vec_ic1.append(model1.intercept_)
        
        yfit1 = model1.predict(np.sort(nga_time1).reshape(-1, 1))

        #plt.plot([p for p, _ in yfit1], [q for _, q in yfit1])
        
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

        yfit1 = model1.predict(np.sort(nga_time1).reshape(-1, 1))
        yfit2 = model2.predict(np.sort(nga_time2).reshape(-1, 1))
        
        #plt.plot([p for p, _ in yfit1], [q for _, q in yfit1])
        #plt.plot([p for p, _ in yfit2], [q for _, q in yfit2])

        vec_coef1.append(model1.coef_)
        vec_ic1.append(model1.intercept_)
        vec_coef2.append(model2.coef_)
        vec_ic2.append(model2.intercept_)

    #plt.scatter(nga_pc1, nga_pc2, s=9, c=range(len(nga_pc1)), cmap = 'Blues')
    #plt.show()
    #plt.close()
    
gauss1_coef = np.mean(vec_coef1, axis=0)
gauss1_ic = np.mean(vec_ic1, axis=0)
gauss2_coef = np.mean(vec_coef2, axis=0)
gauss2_ic = np.mean(vec_ic2, axis=0)
