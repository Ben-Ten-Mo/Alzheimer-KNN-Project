# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import pandas as pd
# To make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import seaborn as sns
#set style of plots
sns.set_style('white')

#define a custom palette
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
sns.set_palette(customPalette)
sns.palplot(customPalette)

#Shows the Color Palette
#plt.show()

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

patients = pd.read_csv("../Downloads/alzheimer.csv")
# 'Group','M/F',
patients = patients.drop(['MMSE','CDR','eTIV','nWBV','ASF'],axis=1)
mapping_MF = {'M': 0, 'F': 1}
mapping_Dem = {'Nondemented': 0, 'Converted': 0.5, 'Demented': 1}
patients = patients.replace({'Group': mapping_Dem, 'M/F': mapping_MF})
#patients = patients.dropna()
print(patients.describe())
from sklearn import preprocessing

age = patients[['Age']].values
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
patients['Age'] = pd.DataFrame(age_scaled)

education = patients[['EDUC']].values
education_scaled = min_max_scaler.fit_transform(education)
patients['EDUC'] = pd.DataFrame(education_scaled)

ses = patients[['SES']].values
ses_scaled = min_max_scaler.fit_transform(ses)
patients['SES'] = pd.DataFrame(ses_scaled)

patients = patients.dropna()

patients.hist(bins=50,figsize=(20,15))

plt.show()

#Doesn't work because of missing values and might need to convert M/F t 0 and 1 same for groups

patient_aspects = patients.copy()
print(patient_aspects)
from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(patient_aspects)
    Sum_of_squared_distances.append(km.inertia_)

from sklearn.metrics import silhouette_score
for n_clusters in range(2,15):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(patient_aspects)
    centers = clusterer.cluster_centers_

    score = silhouette_score (patient_aspects, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

plt.plot(K, Sum_of_squared_distances, 'gx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(patient_aspects)

from sklearn.decomposition import PCA
y_kmeans = kmeans.predict(patient_aspects)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(patient_aspects)

pc = pd.DataFrame(principal_components)
pc['label'] = y_kmeans
pc.columns = ['x', 'y','label']

#plot data with seaborn
sns.lmplot(data=pc, x='x', y='y', hue='label', 
                   fit_reg=False, legend=True)
#plt.show()

from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
#T-SNE with two dimensions
tsne = TSNE(n_components=2, perplexity=50)

tsne_components = tsne.fit_transform(patient_aspects)


ts = pd.DataFrame(tsne_components)
ts['label'] = y_kmeans
ts.columns = ['x', 'y','label']

#plot data with seaborn
cluster = sns.lmplot(data=ts, x='x', y='y', hue='label', 
                   fit_reg=False, legend=True)

print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

# Dump components relations with features:
print (pd.DataFrame(pca.components_,columns=patient_aspects.columns,index = ['PC-1','PC-2']))

patients['label'] = y_kmeans

# shuffle dataset

patients = patients.sample(frac=1)
print(patients['label'].value_counts())

print(patients[patients['label'] == 0].tail(10))
print(patients[patients['label'] == 0].mean())
print(patients[patients['label'] == 1].tail(10))
print(patients[patients['label'] == 1].mean())
print(patients[patients['label'] == 2].tail(10))
print(patients[patients['label'] == 2].mean())
print(patients[patients['label'] == 3].tail(10))
print(patients[patients['label'] == 3].mean())
patients[patients['label']==0].hist()
patients[patients['label']==1].hist()
patients[patients['label']==2].hist()
patients[patients['label']==3].hist()


plt.show()
