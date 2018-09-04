import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt, cm as cm, mlab as mlab
import matplotlib
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set()
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
import os
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

PC1_array = StandardScaler().fit_transform(PC1_df.loc[:, [impute[i]]].values)
# filtered_pnas1 = pnas_CC.loc[:, cc_names].values
#num_bins = 25
num_bins = 50

# the histogram/gaussian mixture model of the data
n, bins, patches = plt.hist(PC1_array, num_bins, normed=1, facecolor='blue', alpha=0.5)
 
plt.title(str(i+1)+"th imputed dataset")
plt.xlabel("Projection onto first Principle Component")
plt.legend()
plt.savefig("pc1_histogram.png")
plt.savefig("pc1_histogram.eps")


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
regionDict["Oceania-Australia"] = ["Oro, PNG","Chuuk Islands","Big Island Hawaii"]

worldRegions = list(regionDict.keys())

#for region in worldRegions:
#    for nga in regionDict[region]

#9-d interpolation and scatter plot by NGAs

# interpolating on the 1-d principal axis
nga_interpolated = defaultdict(list)

for impute_index in range(1, numImpute+1):
    # 1) polity-based interpolation
    impute_set = CC_df[CC_df.irep==impute_index]
    unique_regions = impute_set.NGA.unique().tolist()

    for nga in unique_regions:
        times = sorted(impute_set[impute_set.NGA == nga].Time.unique().tolist())
        data_idx = list()
        for t in times: 
            data_idx.append(CC_df.index[(CC_df['NGA'] == nga) &
                                             (CC_df['Time'] == t) &
                                             (CC_df['irep'] == impute_index)].tolist()[0])
        nga_interpolated[nga].extend([(PC_matrix[:,0][idx], time) for idx, time in zip(data_idx, times)])
        
        if len(times) != ((max(times)-min(times))/100)+1:
            for time in range(len(times)-1):
                if times[time+1]-times[time] != 100:
                    # linear interpolation
                    val1_idx = CC_df.index[(CC_df['NGA'] == nga) & 
                                                (CC_df['Time'] == times[time]) &
                                                (CC_df['irep'] == impute_index)].tolist()[0]
                    val2_idx = CC_df.index[(CC_df['NGA'] == nga) & 
                                                (CC_df['Time'] == times[time+1]) &
                                               (CC_df['irep'] == impute_index)].tolist()[0]

                    diff = PC_matrix[:,0][val2_idx] - PC_matrix[:,0][val1_idx]
                    
                    num_steps = int((times[time+1]-times[time])/100)

                    for i in range(1, num_steps):
                        diff_step = (i/num_steps)*diff
                        interpol = diff_step+PC_matrix[:,0][val1_idx]
                        nga_interpolated[nga].append((interpol, times[time]+100*i))
    
colors = iter(cm.rainbow(np.linspace(0, 1, len(NGAs))))
for nga in NGAs:
    period = [time for val, time in nga_interpolated[nga]]
    pc_proj = [val for val, time in nga_interpolated[nga]]
    plt.scatter(period, pc_proj, s=1.5, color=next(colors))
plt.title('interpolated')
plt.xlabel('time period')
plt.ylabel('pc projection value')
plt.savefig('interpolated_scatter.png', transparent=True)
plt.show()
plt.close()
