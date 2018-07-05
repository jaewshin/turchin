import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy 

pnas_data1 = pd.read_csv('pnas_data1.csv')
pnas_data2 = pd.read_csv('pnas_data2.csv')
raw = pd.read_csv('seshat-20180402.csv', encoding = 'ISO-8859-1')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features = ['PolPop', 'PolTerr', 'CapPop', 'levels', 'government','infrastr', 'writing', 'texts', 'money']

# for i in range(20):
# 	imputed_set = pnas_data1[pnas_data1.irep == (i+1)]
# 	X = imputed_set.loc[:, features].values

# 	X = StandardScaler().fit_transform(X)
# 	pca = PCA(n_components=9)
# 	pca.fit(X)
# 	print(str(i+1) + 'th imputed set', pca.explained_variance_)


X = pnas_data1.loc[:, features].values
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)
X_new = pca.inverse_transform(X_pca)


# X = pnas_data1.loc[:, features].values
# X = StandardScaler().fit_transform(X)
# pca = PCA(n_components=9)
# # pca.fit(X)

# print(pca.components_)
# print(pca.explained_variance_)


# 3d plot with 3 PCs
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')




# X, Y, Z = X_new[:, 0], X_new[:, 1], X_new[:, 2]
# ax.scatter(X, Y, Z)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()


#histogram definition
xyrange = [[-0.5,2],[-0.5,2]] # data range
bins = [100,100] # number of bins
thresh = 3  #density threshold

#data definition
N = 1e5;
X, Y= X_new[:, 0], X_new[:, 1]

# histogram the data
hh, locx, locy = scipy.histogram2d(X, Y, range=xyrange, bins=bins)
posx = np.digitize(X, locx)
posy = np.digitize(Y, locy)

#select points within the histogram
ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
xdat1 = X[ind][hhsub < thresh] # low density points
ydat1 = Y[ind][hhsub < thresh]
hh[hh < thresh] = np.nan # fill the areas with low density by NaNs

plt.imshow(np.flipud(hh.T),cmap='jet',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper')
plt.colorbar()   
plt.plot(xdat1, ydat1, '.',color='darkblue')
plt.show()	