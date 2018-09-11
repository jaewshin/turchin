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
n, bins, patches = plt.hist(PC_matrix[:,0], num_bins, normed=1, facecolor='blue', alpha=0.5)
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
