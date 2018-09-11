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

# Plot PC1 by time for each region
# First, map from World Region to Late, Intermediate, and Early NGAs
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

worldRegions = list(regionDict.keys())

#xmin = min(CC_df['Time'])
#xmax = max(CC_df['Time'])
xmin = -10000
#xmin = -4000
xmax = 2000
#ymin = min(PC_matrix[:,0])
#ymax = max(PC_matrix[:,0])
ymin = -7
ymax = 7
#f, axes = plt.subplots(int(len(worldRegions)/2),1, sharex=True, sharey=True,figsize=(3,10))
#f, axes = plt.subplots(len(worldRegions),1, sharex=True, sharey=True,figsize=(1,10))
f, axes = plt.subplots(int(len(worldRegions)/2),2, sharex=True, sharey=True,figsize=(12,15))
axes[0,0].set_xlim([xmin,xmax])
axes[0,0].set_ylim([ymin,ymax])
#axes[0].xlabel("Calendar Date [AD]")
#axes[0].ylabel("PC1")
for i,reg in enumerate(worldRegions):
    regList = list(reversed(regionDict[reg]))
    #plotNum = math.floor(i/2)
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
    #axes[plotNum].set_title(reg,fontsize=10)
    s = '{' + regList[0] + '; ' + regList[1] + '; ' + regList[2] + '}'
    axes[m,n].set_title(s,fontsize=10)
    if m != 4:
        plt.setp(axes[m,n].get_xticklabels(), visible=False)
        #axes[plotNum].scatter(CC_df.loc[ind,'Time'],PC_matrix[ind,0],alpha=.4,s=1.5)
    #axes[plotNum].legend(regList,bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
    #axes[plotNum].legend(regList,bbox_to_anchor=(0,1.02,1,.102),loc=3,ncol=3,mode='expand',borderaxespad=0)
        #axes[i].xlim(xmin,xmax)
    #plt.ylim(ymin,ymax)
    #if i % 2 == 1:
    #    axes[plotNum].legend(regListPrev + regList,bbox_to_anchor=(1, 0.5))
    #regListPrev = regList
f.subplots_adjust(hspace=.5)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.savefig("pc1_vs_time_stacked_by_region.pdf")
plt.savefig("pc1_vs_time_stacked_by_region.eps")
plt.savefig("pc1_vs_time_stacked_by_region.png")
plt.close()

early = [regionDict[reg][2] for reg in worldRegions]
middle = [regionDict[reg][1] for reg in worldRegions]
late = [regionDict[reg][0] for reg in worldRegions]

allStarts = (early,middle,late)
startString = ['Early','Middle','Late']

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
#f.subplots_adjust(hspace=.5)
#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.savefig("pc1_vs_time_stacked_by_start.pdf")
plt.savefig("pc1_vs_time_stacked_by_start.eps")
plt.savefig("pc1_vs_time_stacked_by_start.png")
plt.close()
            

##for region in worldRegions:
##    for nga in regionDict[region]
#
##9-d interpolation and scatter plot by NGAs
#
## interpolating on the 1-d principal axis
#nga_interpolated = defaultdict(list)
#
#for impute_index in range(1, numImpute+1):
#    # 1) polity-based interpolation
#    impute_set = CC_df[CC_df.irep==impute_index]
#    unique_regions = impute_set.NGA.unique().tolist()
#
#    for nga in unique_regions:
#        times = sorted(impute_set[impute_set.NGA == nga].Time.unique().tolist())
#        data_idx = list()
#        for t in times: 
#            data_idx.append(CC_df.index[(CC_df['NGA'] == nga) &
#                                             (CC_df['Time'] == t) &
#                                             (CC_df['irep'] == impute_index)].tolist()[0])
#        nga_interpolated[nga].extend([(PC_matrix[:,0][idx], time) for idx, time in zip(data_idx, times)])
#        
#        if len(times) != ((max(times)-min(times))/100)+1:
#            for time in range(len(times)-1):
#                if times[time+1]-times[time] != 100:
#                    # linear interpolation
#                    val1_idx = CC_df.index[(CC_df['NGA'] == nga) & 
#                                                (CC_df['Time'] == times[time]) &
#                                                (CC_df['irep'] == impute_index)].tolist()[0]
#                    val2_idx = CC_df.index[(CC_df['NGA'] == nga) & 
#                                                (CC_df['Time'] == times[time+1]) &
#                                               (CC_df['irep'] == impute_index)].tolist()[0]
#
#                    diff = PC_matrix[:,0][val2_idx] - PC_matrix[:,0][val1_idx]
#                    
#                    num_steps = int((times[time+1]-times[time])/100)
#
#                    for i in range(1, num_steps):
#                        diff_step = (i/num_steps)*diff
#                        interpol = diff_step+PC_matrix[:,0][val1_idx]
#                        nga_interpolated[nga].append((interpol, times[time]+100*i))
#    
#colors = iter(cm.rainbow(np.linspace(0, 1, len(NGAs))))
#for nga in NGAs:
#    period = [time for val, time in nga_interpolated[nga]]
#    pc_proj = [val for val, time in nga_interpolated[nga]]
#    plt.scatter(period, pc_proj, s=1.5, color=next(colors))
#plt.title('interpolated')
#plt.xlabel('time period')
#plt.ylabel('pc projection value')
#plt.savefig('interpolated_scatter.png', transparent=True)
#plt.show()
#plt.close()
