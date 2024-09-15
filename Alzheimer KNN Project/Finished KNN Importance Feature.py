from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

#from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt



patients = pd.read_csv("../Downloads/alzheimer.csv")
# 'Group','M/F',
patients = patients.drop(['SES','CDR'],axis=1)
patients.head()
mapping_MF = {'M': 0, 'F': 1}
patients = patients.replace({ 'M/F': mapping_MF})
patients = patients.dropna()

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

from sklearn.neighbors import KNeighborsClassifier

#from IPython.display import display

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

dep_var = "Demented"
cond = np.random.rand(len(patients))>.2
train = np.where(cond)[0]
valid = np.where(~cond)[0]

len(train), len(valid)

train_df = patients.iloc[train]
valid_df = patients.iloc[valid]
len(train_df),len(valid_df)

train_y = train_df['Group']
train_xs = train_df.drop(['Group'],axis=1)

valid_y = valid_df['Group']
valid_xs = valid_df.drop(['Group'],axis=1)

m = KNeighborsClassifier()
m = m.fit(train_xs,train_y)

plot_confusion_matrix(estimator=m,X=valid_xs,y_true=valid_y,cmap=plt.cm.Greens)


std_score = m.score(valid_xs,valid_y)
print(std_score)

data = {'Age':[0],'EDUC':[0],'MMSE':[0],'eTIV':[0],'nWBV':[0],'ASF':[0],'M/F':[0]}
feat_imp = pd.DataFrame(data)
print(feat_imp.head())

#Repeat this section: AGE
print(valid_xs.head())

valid_Age = valid_xs.copy()
valid_Age['Age'] = np.random.permutation(valid_Age['Age'])
valid_Age.head()
m.score(valid_Age,valid_y)

feat_imp['Age'] = std_score - m.score(valid_Age,valid_y)
print(feat_imp.head())
#plt.show()

# EDUC

valid_xs.head()

valid_EDUC = valid_xs.copy()
valid_EDUC['EDUC'] = np.random.permutation(valid_EDUC['EDUC'])
valid_EDUC.head()
m.score(valid_EDUC,valid_y)

feat_imp['EDUC'] = std_score - m.score(valid_EDUC,valid_y)
print(feat_imp.head())
#MMSE
valid_xs.head()

valid_MMSE = valid_xs.copy()
valid_MMSE['MMSE'] = np.random.permutation(valid_MMSE['MMSE'])
valid_MMSE.head()
m.score(valid_MMSE,valid_y)

feat_imp['MMSE'] = std_score - m.score(valid_MMSE,valid_y)
print(feat_imp.head())
#eTIV
valid_xs.head()

valid_eTIV = valid_xs.copy()
valid_eTIV['eTIV'] = np.random.permutation(valid_eTIV['eTIV'])
valid_eTIV.head()
m.score(valid_eTIV,valid_y)

feat_imp['eTIV'] = std_score - m.score(valid_eTIV,valid_y)
print(feat_imp.head())
#nWBV
valid_xs.head()

valid_nWBV = valid_xs.copy()
valid_nWBV['nWBV'] = np.random.permutation(valid_nWBV['nWBV'])
valid_nWBV.head()
m.score(valid_nWBV,valid_y)

feat_imp['nWBV'] = std_score - m.score(valid_nWBV,valid_y)
print(feat_imp.head())
#ASF
valid_xs.head()

valid_ASF = valid_xs.copy()
valid_ASF['ASF'] = np.random.permutation(valid_ASF['ASF'])
valid_ASF.head()
m.score(valid_ASF,valid_y)

feat_imp['ASF'] = std_score - m.score(valid_ASF,valid_y)
print(feat_imp.head())
#M/F
valid_xs.head()

valid_MF = valid_xs.copy()
valid_MF['M/F'] = np.random.permutation(valid_MF['M/F'])
valid_MF.head()
m.score(valid_MF,valid_y)

feat_imp['M/F'] = std_score - m.score(valid_MF,valid_y)
print(feat_imp.head())

plt.show()
