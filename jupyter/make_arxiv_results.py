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

def draw_vector(v0, v1, ax=None, description = ''):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate(description, v1, v0, arrowprops=arrowprops)

times = CC_df.loc[:, ['Time']].values

#fit GMM
gmm = GMM(n_components=2).fit(PC_matrix)
cov = gmm.covariances_
prob_distr = gmm.predict_proba(PC_matrix)

# determine to which of the two gaussians each data point belongs by looking at probability distribution 
gauss1_idx = [i for i in range(len(prob_distr)) if prob_distr[i][0] >= prob_distr[i][1]]
gauss2_idx = [j for j in range(len(prob_distr)) if prob_distr[j][1] >= prob_distr[j][0]]

gauss1_time = [times[i] for i in gauss1_idx] # time for the first gaussian data
gauss2_time = [times[j] for j in gauss2_idx] # time for the second gaussian data

gauss1_pc1 = [PC_matrix[:,0][i] for i in gauss1_idx] # first pc values for the first gaussian
gauss2_pc1 = [PC_matrix[:,0][j] for j in gauss2_idx] # first pc values for the second gaussian

gauss1_pc2 = [PC_matrix[:,1][i] for i in gauss1_idx]
gauss2_pc2 = [PC_matrix[:,1][j] for j in gauss2_idx]

plt.scatter(gauss1_pc1, gauss1_pc2, s=3, c='b')
plt.scatter(gauss2_pc1, gauss2_pc2, s=3, c='r')

# X, Y = data[:, 0], data[:, 1]

# plt.scatter(X, Y, s=3)
# plt.title('scatter plot for two principal component values using all 20 imputed sets')
plt.xlabel('First PC')
plt.ylabel('Second PC')

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

plt.savefig('two_pc.png', transparent=True)
v0 = np.asarray([gauss1_x[0], gauss1_y[0]]); v1 = np.asarray([gauss1_x[-1], gauss1_y[-1]])
draw_vector(v0, v1)
v0 = np.asarray([gauss2_x[0], gauss2_y[0]]); v1 = np.asarray([gauss2_x[-1], gauss2_y[-1]])
draw_vector(v0, v1)
plt.savefig('dynamics.png', transparent=True)
#plt.show()
plt.close()

plt.scatter(gauss1_pc1, gauss1_pc2, s=3, c='b')
plt.scatter(gauss2_pc1, gauss2_pc2, s=3, c='r')

# X, Y = data[:, 0], data[:, 1]

# plt.scatter(X, Y, s=3)
# plt.title('scatter plot for two principal component values using all 20 imputed sets')
plt.xlabel('First PC')
plt.ylabel('Second PC')

gauss1_t = sorted(gauss1_time)
gauss2_t = sorted(gauss2_time)
ic1 = np.asarray([gauss1_ic]).T
ic2 = np.asarray([gauss2_ic]).T

#def line_vec(time, coef, intercept):
#    return [(coef*i+intercept) for i in time]

gauss1 = line_vec(gauss1_t, gauss1_coef, ic1)
gauss2 = line_vec(gauss2_t, gauss2_coef, ic2)

gauss1_x= [i for i,j in gauss1 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]
gauss1_y= [j for i,j in gauss1 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]
gauss2_x= [i for i,j in gauss2 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]
gauss2_y= [j for i,j in gauss2 if (i<=4 and i>=-6) and (j>=-3 and j<=3)]

plt.savefig('two_pc.png', transparent=True)
v0 = np.asarray([gauss1_x[0], gauss1_y[0]]); v1 = np.asarray([gauss1_x[-1], gauss1_y[-1]])
draw_vector(v0, v1)
v0 = np.asarray([gauss2_x[0], gauss2_y[0]]); v1 = np.asarray([gauss2_x[-1], gauss2_y[-1]])
draw_vector(v0, v1)
plt.savefig('dynamics.png', transparent=True)
#plt.show()
plt.close()

# flow vectors for each NGAs
nga_dict = {key:list() for key in NGAs}
vec_coef1 = list() # coefficients for overall flow vectors for each ngas in the first gaussian
vec_ic1 = list() # intercept for overall flow vectors for each ngas in the first gaussian
vec_coef2 = list() # coefficients for overall flow vectors for each ngas in the second gaussian
vec_ic2 = list() # intercept for overall flow vectors for each ngas in the second gaussian

for idx in range(len(PC_matrix)):
    nga = CC_df.loc[idx].NGA
    nga_dict[nga].append((PC_matrix[:,0][idx], PC_matrix[:,1][idx], times[idx][0], idx))            

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

print(slope1); print(slope2)
print(intercept1); print(intercept2)
plt.scatter(gauss1_pc1, gauss1_pc2, s=3, c='b')
plt.scatter(gauss2_pc1, gauss2_pc2, s=3, c='r')

# plot flow vectors 
def line_vec2(xlinspace, slope, intercept):
    return [(i, slope*i+intercept) for i in xlinspace]

gauss1 = line_vec2(np.linspace(-6, 0), slope1, intercept1)
gauss2 = line_vec2(np.linspace(-1, 4), slope2, intercept2)

gauss1_x, gauss1_y = zip(*gauss1)
gauss2_x, gauss2_y = zip(*gauss2)

v0 = np.asarray([gauss1_x[0], gauss1_y[0]]); v1 = np.asarray([gauss1_x[-1], gauss1_y[-1]])
draw_vector(v0, v1)
v0 = np.asarray([gauss2_x[0], gauss2_y[0]]); v1 = np.asarray([gauss2_x[-1], gauss2_y[-1]])
draw_vector(v0, v1)


plt.xlabel('First PC')
plt.ylabel('Second PC')
#plt.show()
plt.close()

# llinear discriminant analysis 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = LinearDiscriminantAnalysis(n_components = 9)

gmm = GMM(n_components=2).fit(CC_scaled)
cov = gmm.covariances_
prob_distr = gmm.predict_proba(CC_scaled)
gauss_idx = [1 if prob_distr[i][0] >= prob_distr[i][1] else 0 for i in range(len(prob_distr))]

clf.fit(CC_scaled, gauss_idx)
print(clf.coef_)
print(clf.intercept_)



# what contributes to the angle 
# MHP: Where does y come from on the next line?
y = [0.06630305, 0.0248204, 0.08419158, 0.2790867, 0.01331484, 0.08750921, 0.01332944, 0.07048244, 0.25338994]
y = [i/sum(y) * 100 for i in y]
assert sum(y) == 100
x = range(1, 10)

num_bins = 9

plt.plot(x, y)
plt.xlabel('Principal Components')
plt.ylabel('Contribution to the angle in percentages')
plt.savefig('angle plot', transparent=True)
#plt.show()
plt.close()

print(np.corrcoef(CC_scaled, rowvar = False)) # all the variables are highly correlated

# flow vectors between the two clusters 

# Per Python scoping, variables inside the following function are found in the global environment
# Maybe we should make these variables inputs to the function?
def flowvec_inbetween():
    #Gaussian Mixture Model 
    #fit GMM
    gmm = GMM(n_components=2).fit(PC_matrix)
    cov = gmm.covariances_
    prob_distr = gmm.predict_proba(PC_matrix)
    
    
    # determine to which of the two gaussians each data point belongs by looking at probability distribution 
    gauss_idx1 = [i for i in range(len(prob_distr)) if prob_distr[i][0] <= 0.9]
    gauss_idx2 = [i for i in range(len(prob_distr)) if prob_distr[i][1] <= 0.9]
    print(len(gauss_idx1))
    print(len(gauss_idx2))
    print(len(list(set(gauss_idx1) & set(gauss_idx2))))
    
    print(len([i for i in prob_distr if i[0] < i[1]]))
    
    # flow vectors for each NGAs
    nga_dict = {key:list() for key in NGAs}
    vec_coef = list() # coefficients for overall flow vectors for each ngas between the two gaussian
    vec_ic = list() # intercept for overall flow vectors for each ngas between the two gaussian

    for idx in range(len(PC_matrix)):
        nga = CC_df.loc[idx].NGA
        nga_dict[nga].append((PC_matrix[:,0][idx], PC_matrix[:,1][idx], times[idx][0], idx))            

    for i in range(len(NGAs)):  # flow vector for each NGA

        nga_pc = [[p,j] for p,j,_,t in nga_dict[NGAs[i]] if t in gauss_idx]
        nga_time = np.asarray([k for _,_,k,t in nga_dict[NGAs[i]] if t in gauss_idx])


        #fit linear regression 
        if len(nga_pc) != 0:
            ols = linear_model.LinearRegression()
            model = ols.fit(nga_time.reshape(-1,1), nga_pc)

            vec_coef.append(model.coef_)
            vec_ic.append(model.intercept_)
    
    return vec_coef, vec_ic

def calc_slope(vec_coef, vec_ic):
    # slope and intercept of the average flow vector for each cluster
    np.seterr(divide='ignore', invalid='ignore')
    
    gauss_coef = np.mean(vec_coef, axis=0)
    gauss_ic = np.mean(vec_ic, axis=0)

    slope = gauss_coef[1]/gauss_coef[0]
    intercept = gauss_ic[1] - ((slope) * gauss_ic[0])
    
    return slope[0], intercept[0]

# plt.scatter(gauss1_pc1, gauss1_pc2, s=3, c='b')
# plt.scatter(gauss2_pc1, gauss2_pc2, s=3, c='r')

# # plot flow vectors 
# def line_vec(xlinspace, slope, intercept):
#     return [(i, slope*i+intercept) for i in xlinspace]

# gauss = line_vec(np.linspace(-3, 2), slope, intercept)

# gauss1_x, gauss1_y = zip(*gauss1)
# gauss2_x, gauss2_y = zip(*gauss2)

# v0 = np.asarray([gauss1_x[0], gauss1_y[0]]); v1 = np.asarray([gauss1_x[-1], gauss1_y[-1]])
# draw_vector(v0, v1)
# v0 = np.asarray([gauss2_x[0], gauss2_y[0]]); v1 = np.asarray([gauss2_x[-1], gauss2_y[-1]])
# draw_vector(v0, v1)


# plt.xlabel('First PC')
# plt.ylabel('Second PC')
# plt.show()
# plt.close()

# resampling from flow vector of each NGAs 

#pt = turchin.progress_timer(n_iter = 5000, description = 'average flow vector bootstrapping on NGAs')

# flow vectors for each NGAs
vec_coef, vec_ic = flowvec_inbetween()
orig_slope, orig_ic = calc_slope(vec_coef, vec_ic)

def bstr_nga_flow(vec_coef, vec_ic, n=5000):
    """
    Resample from the flow vector of each nga with respect to each clusters
    """
    slopes = []
    intercepts = []
    
    vec_coef = np.asarray(vec_coef)
    vec_ic = np.asarray(vec_ic) 
    
    for i in range(n):
        
        #resample
        resample = np.random.randint(0, len(vec_coef), size = len(vec_coef))
        resampled_coef = vec_coef[resample]
        resampled_ic = vec_ic[resample]
        
        # find the average flow vector for each cluster
        slope, ic = calc_slope(resampled_coef, resampled_ic)
        
        if not np.isnan(slope):
            slopes.append(slope)
            intercepts.append(ic)
        
        #pt.update()
        
    return slopes, intercepts 

slopes, intercepts = bstr_nga_flow(vec_coef, vec_ic)
#pt.finish()

# plot the histogram 
num_bins = 100

print(len(slopes))
# average flow vector for the first Gaussian after bootstrapping
n, bins, patches = plt.hist(slopes, num_bins, facecolor='blue', alpha=0.5)
plt.title('average slope of the flow vector between the two clusters')
plt.axvline(x=orig_slope, c= 'Black')
plt.xlabel('slope')
plt.ylabel('number of occurences')
#plt.show()
plt.close()

# average flow vector for the second Gaussian after bootstrapping 
n, bins, patches = plt.hist(intercepts, num_bins, facecolor='blue', alpha=0.5)
plt.title('average intercept of the flow vector between the two clusters')
plt.axvline(x=orig_ic, c='Black')
plt.xlabel('intercept')
plt.ylabel('number of occurences')
#plt.show()
plt.close()

# find the data points within each NGA where each point has less than 90% of belonging to either of the clusters 

def flowvec_points(threshold):
    """
    Return the data points within each NGA where each point has less than 90% of belonging to either of the clusters
    """
    gmm = GMM(n_components=2).fit(PC_matrix)
    cov = gmm.covariances_
    prob_distr = gmm.predict_proba(PC_matrix)
    
    # determine to which of the two gaussians each data point belongs by looking at probability distribution 
    gauss_idx = [i for i in range(len(prob_distr)) 
                  if (prob_distr[i][0] <= threshold and prob_distr[i][1] <= threshold)]
    return gauss_idx

# flowvec_points(0.9)


idx_len = []
range_val = []
threshold = '0.9'

for i in range(100):
    thres = float(threshold)
    idx = flowvec_points(thres)
        
    idx_len.append(len(idx))
    range_val.append(thres)
    
    if len(idx) == 8280:
        break

    threshold += '9'
    
from math import *

plt.plot(range(len(range_val)), idx_len)
plt.xlabel('number of digits for threshold')
plt.ylabel('number of points included')
#plt.show()
plt.close()

plt.plot([i*100 for i in range_val], idx_len)
plt.xlabel('percentage of threshold')
plt.ylabel('number of points included')
#plt.show()
plt.close()

# find the transition point for each NGAs 

# Best to use a unique name for the function since gauss_idx is a variable above
def gauss_idx_func(CC_scaled):
    """
    Label each point by which of the two Gaussians it has higher probability of belonging to
    """
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
        
    return gauss1_idx, gauss2_idx 

def rand_impute(orig_dataframe):
    """
    Given a set of 20 imputed sets combined, randomly sample to construct 1 set of data points for each polity/time
    """
    
    resample = [np.random.randint(0, 20) * 414 + i for i in range(414)]
    resampled_df = orig_dataframe.loc[resample]
    
    return resampled_df    
    
def between_vec(df, switch):
    """
    Flow vectors between the two clusters for each NGA
    """    
    gauss1_idx, gauss2_idx = gauss_idx_func(CC_scaled)
    nga_dict = {key:list() for key in NGAs}
    
    slopes = [] 
    
    def slope(a, b):
        """ find slope given two points """
        a1, a2 = PC_matrix[:, 0][a], PC_matrix[:, 1][a]
        b1, b2 = PC_matrix[:, 0][b], PC_matrix[:, 1][b]
        
        return b1-a1, b2-a2
    
    # compute flow vector for each nga 
    for nga in NGAs:
        nga_idx = df.index[df['NGA'] == nga].tolist()

        gauss1 = [i for i in nga_idx if i in gauss1_idx]
        gauss2 = [j for j in nga_idx if j in gauss2_idx]

        # use the last point in the first cluster and the first point in the second cluster
        if switch == 1: 
            
            try:
                a, b = gauss1[-1], gauss2[0]
                x, y = slope(a, b)
                slopes.append((x, y))

            except: # lies only in one of the two clusters 
                pass 
        
        # use the very first time points make a transition from the first to the second
        elif switch == 2:
            
            for idx in range(len(nga_idx)-1):
                
                if nga_idx[idx] in gauss1 and nga_idx[idx+1] in gauss2:
                    
                    a, b = nga_idx[idx], nga_idx[idx+1]
                    x, y = slope(a, b)
                    slopes.append((x, y))
                    
                    break 
        
        # take all transitions
        elif switch == 3:
            
            for idx in range(len(nga_idx)-1):
                
                if nga_idx[idx] in gauss1 and nga_idx[idx+1] in gauss2:
                    
                    a, b = nga_idx[idx], nga_idx[idx+1]
                    x, y = slope(a, b)
                    slopes.append((x, y))
        
    return slopes 

from math import * 

def std(mean, vals):
    """
    Compute the standard deviation given the mean and a list of values
    """
    return sqrt(sum([(i-mean)**2 for i in vals])/len(vals))


df = rand_impute(CC_df)

for i in [1, 2, 3]:
    slopes = between_vec(df, i)

    x_coor = [i for i, _ in slopes]
    y_coor = [j for _, j in slopes]
    mean_x = sum(x_coor)/len(x_coor)
    mean_y = sum(y_coor)/len(y_coor)
    
    std_x = std(mean_x, x_coor)/sqrt(len(x_coor)-1)
    std_y = std(mean_y, y_coor)/sqrt(len(y_coor)-1)

    print("mean", mean_x, mean_y)
    print("errors", std_x, std_y)
    print("first component error bar start", mean_x-std_x, "error bar end", mean_x+std_x)
    print("second component error bar start", mean_y-std_y, "error bar end", mean_y+std_y)
    print("\n\n")

 # from functools import reduce

# avg_slope_ = np.average(np.asarray(slopes))
# print(avg_slope_)
# simple bootstrap 
# pt = turchin.progress_timer(n_iter = 5000, description = 'average slope for flow vectors inbetween')

def bstr(slopes, n=5000):
    """
    Given data matrix, perform bootstrapping by collecting n samples (default = 5000) and return the 
    error rate for the mean of the data. Assume that the given data matrix is numpy array
    """            
    resample_avg = list() # resampling of avg of the slopes 
    
    for i in range(n):
        
        resample = np.random.randint(0, len(slopes), size=len(slopes))
        resampled = PC_matrix[resample]
        resample_avg.append(np.average(resampled))
        
        pt.update()
        
    return resample_avg 

# avg_slope = bstr(slopes)
# pt.finish()

# # plot the histogram for eigenvalues and angles
# num_bins = 200
# # angle between PC and the first component
# n, bins, patches = plt.hist(avg_slope, num_bins, facecolor='blue', alpha=0.5)
# plt.title('average slope of the flow vector between two clusters')
# plt.axvline(x=avg_slope_)
# plt.show()
# plt.close()

    
#     # flow vectors for each NGAs
#     nga_dict = {key:list() for key in ngas}
#     vec_coef = list() # coefficients for overall flow vectors for each ngas between the two gaussian
#     vec_ic = list() # intercept for overall flow vectors for each ngas between the two gaussian

#     for idx in range(len(data)):
#         nga = pnas_data1.loc[idx].NGA
#         nga_dict[nga].append((data[:,0][idx], data[:,1][idx], times[idx][0], idx))            

#     for i in range(len(ngas)):  # flow vector for each NGA

#         nga_pc = [[p,j] for p,j,_,t in nga_dict[ngas[i]] if t in gauss_idx]
#         nga_time = np.asarray([k for _,_,k,t in nga_dict[ngas[i]] if t in gauss_idx])

#         #fit linear regression 
#         if len(nga_pc) != 0:
#             ols = linear_model.LinearRegression()
#             model = ols.fit(nga_time.reshape(-1,1), nga_pc)

#             vec_coef.append(model.coef_)
#             vec_ic.append(model.intercept_)
    
#     return vec_coef, vec_ic
