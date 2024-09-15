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
patients = patients.drop(['SES','CDR'],axis=1)
mapping_MF = {'M': 0, 'F': 1}
mapping_Dem = {'Nondemented': 0, 'Converted': 0.5, 'Demented': 1}
patients = patients.replace({'Group': mapping_Dem, 'M/F': mapping_MF})
patients = patients.dropna()
print(patients.describe())
from sklearn import preprocessing

age = patients[['Age']].values
min_max_scaler = preprocessing.MinMaxScaler()
age_scaled = min_max_scaler.fit_transform(age)
patients['Age'] = pd.DataFrame(age_scaled)

education = patients[['EDUC']].values
education_scaled = min_max_scaler.fit_transform(education)
patients['EDUC'] = pd.DataFrame(education_scaled)
'''
ses = patients[['SES']].values
ses_scaled = min_max_scaler.fit_transform(ses)
patients['SES'] = pd.DataFrame(ses_scaled)
'''
mental_state = patients[['MMSE']].values
mmse_scaled = min_max_scaler.fit_transform(mental_state)
patients['MMSE'] = pd.DataFrame(mmse_scaled)

intracranial_v = patients[['eTIV']].values
eTIV_scaled = min_max_scaler.fit_transform(intracranial_v)
patients['eTIV'] = pd.DataFrame(eTIV_scaled)

atlas_scaling = patients[['ASF']].values
ASF_scaled = min_max_scaler.fit_transform(atlas_scaling)
patients['ASF'] = pd.DataFrame(ASF_scaled)

patients = patients.dropna()
patients_2 = patients.copy()
patients = patients.drop(['Group'],axis=1)
patients = patients.drop(['M/F'],axis=1)

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
patients_2['label'] = y_kmeans
# shuffle dataset

patients = patients.sample(frac=1)
print(patients['label'].value_counts())

print(patients[patients['label'] == 0].tail(10))
#print(patients[patients['label'] == 0].mean())
print(patients_2[patients_2['label'] == 0].mean())

print(patients[patients['label'] == 1].tail(10))
#print(patients[patients['label'] == 1].mean())
print(patients_2[patients_2['label'] == 1].mean())

print(patients[patients['label'] == 2].tail(10))
#print(patients[patients['label'] == 2].mean())
print(patients_2[patients_2['label'] == 2].mean())

print(patients[patients['label'] == 3].tail(10))
#print(patients[patients['label'] == 3].mean())
print(patients_2[patients_2['label'] == 3].mean())


patients[patients['label']==0].hist()
patients[patients['label']==1].hist()
patients[patients['label']==2].hist()
patients[patients['label']==3].hist()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

#print(patient_aspects)
X = patient_aspects
y = y_kmeans
#y_kmeans = kmeans.predict(patient_aspects)
#print(y_kmeans)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
'''
X_train = patients[patients_2['Group'] != 0.5].drop(['label'],axis=1)
#X_train = X_train.drop(['Group','M/F','label'],axis=1).dropna()
X_test = patients[patients_2['Group'] == 0.5].drop(['label'],axis=1)
#X_test = X_test.drop(['Group','M/F','label'],axis=1).dropna()
y_train = kmeans.predict(X_train)
y_test =  kmeans.predict(X_test)
'''
#Try to manually split my own training and testing dataset
rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
rfc.fit(X_train,y_train)

# Predicting the Test set results
y_pred = rfc.predict(X_test)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
    
# Confusion matrix
#definitions = ['Male Nondemented','Female Nondemented']
definitions = ['Most Likely to Convert','Most Likely Nondemented','Likely Nondemented','Already Demented']

# reversefactor = dict(zip(range(4),definitions))
# actual = np.vectorize(reversefactor.get)(y_test)
# pred = np.vectorize(reversefactor.get)(y_pred)
# print(pd.crosstab(actual, pred, rownames=['Actual Mood'], colnames=['Predicted Mood']))

plot_confusion_matrix(y_test, y_pred, classes=definitions,
                      title='Confusion matrix for Random Forest')

# View a list of the features and their importance scores
features = patient_aspects.columns
print(list(zip(patient_aspects[features], rfc.feature_importances_)))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
knn.fit(X_train,y_train)

knn_pred =knn.predict(X_test)

plot_confusion_matrix(y_test, knn_pred, classes=definitions,
                      title='Confusion matrix for KNN')

#Import svm model
from sklearn import svm
#Create a svm Classifier
svm = svm.SVC(kernel="linear") 

#Train the model using the training sets
svm.fit(X_train, y_train)

#Predict the response for test dataset
svm_pred = svm.predict(X_test)

plot_confusion_matrix(y_test, svm_pred, classes=definitions,
                      title='Confusion matrix for SVM')

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()
mlp.fit(X_train, y_train)

mlp_pred = mlp.predict(X_test)

plot_confusion_matrix(y_test, mlp_pred, classes=definitions,
                      title='Confusion matrix for MLP')
print(classification_report(y_test,mlp_pred,target_names=definitions)+'MLP')

print(classification_report(y_test,svm_pred,target_names=definitions)+'SVM')

print(classification_report(y_test,knn_pred,target_names=definitions)+'KNN')

print(classification_report(y_test,y_pred,target_names=definitions)+'forest')

plt.show()
