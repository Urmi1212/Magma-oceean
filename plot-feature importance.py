# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:36:43 2022

@author: Urmi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 22 01:17:33 2022

@author: Urmi
# """


import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import xgboost
import shap


#input data：data_ is source_data，data_0 is "label = 0" data，data_1 is "label = 1"data,data_2 is"label = 2"data
#data_ =  np.loadtxt('./Data_.txt')
data_0 = np.loadtxt('./Kmeans_Data/_data0.txt')
data_1 = np.loadtxt('./Kmeans_Data/_data1.txt')
data_2 = np.loadtxt('./Kmeans_Data/_data2.txt')

#Disarrange the sample order of the data set
#np.random.shuffle(data_)
np.random.shuffle(data_0)
np.random.shuffle(data_1)
np.random.shuffle(data_2)



#DataMode：Stratified sampling was conducted directly (7:3)
x1 = data_0[0:int(data_0.shape[0]*0.7)] 
x2 = data_1[0:int(data_1.shape[0]*0.7)] 
x3 = data_2[0:int(data_2.shape[0]*0.7)]
Data_Train = np.vstack((x1,x2,x3))

x1 = data_0[int(data_0.shape[0]*0.7):int(data_0.shape[0])] 
x2 = data_1[int(data_1.shape[0]*0.7):int(data_1.shape[0])] 
x3 = data_2[int(data_2.shape[0]*0.7):int(data_2.shape[0])]
Data_Test = np.vstack((x1,x2,x3))

Data_Train_X = Data_Train[:,0:-1]
Data_Train_Y = Data_Train[:,-1]
Data_Test_X = Data_Test[:,0:-1]
Data_Test_Y = Data_Test[:,-1]



# model = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
#                             max_depth=20, max_features='auto', max_leaf_nodes=None,
#                             max_samples=None, min_impurity_decrease=0.0,
#                             min_impurity_split=None, min_samples_leaf=1,
#                             min_samples_split=2, min_weight_fraction_leaf=0.0,
#                             n_estimators=15, n_jobs=None, oob_score=False,
#                             random_state=42, verbose=0, warm_start=False)

    #Random Forest
model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                max_depth=50, max_features=None, max_leaf_nodes=80,
                                max_samples=None, min_impurity_decrease=0.0,
                                min_samples_leaf=1,
                                min_samples_split=2, min_weight_fraction_leaf=0.0,
                                n_estimators=300, n_jobs=-1, oob_score=False,
                                random_state=20, verbose=0, warm_start=False) #min_impurity_split=None, 

# polynomial_features= PolynomialFeatures(degree=2)
# Data_Train_X = polynomial_features.fit_transform(Data_Train_X)

# polynomial_features= PolynomialFeatures(degree=2)
# Data_Test_X = polynomial_features.fit_transform(Data_Test_X)

# model = LinearRegression()
model.fit(Data_Train_X,Data_Train_Y)
 

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(Data_Train_X)

# visualize the first prediction's explanation
#shap.plots.waterfall(shap_values[0])

# summarize the effects of all the features
shap.plots.beeswarm(shap_values)

shap.plots.bar(shap_values)

Data_Train_X_columns = Data_Train_X.columns
print(Data_Train_X_columns.values)

print(model.feature_importances_)
FI = model.feature_importances_
Features = ['T(K)', 'Na2O', 'MgO', 'Al2O3', 'SiO2', 'K2O', 'CaO', 'TiO2', 'MnO', 'FeO', 'Fe2O3']

feature_importance = ({'Features': ['T(K)', 'Na2O', 'MgO', 'Al2O3', 'SiO2', 'K2O', 'CaO', 'TiO2', 'MnO', 'FeO', 'Fe2O3'], 'Feature imp' : [model.feature_importances_]})

print (feature_importance)
import pandas as pd
df1 = pd.DataFrame(feature_importance)

plt.figure(figsize=(10,5))
feature_importance.plt(x="Features", y="Feature imp", kind="bar")

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(Features,FI, color ='maroon')
plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.title("Random Forest")
plt.show()

# Pairplot represents the relationships between different features and also their distributions  
import seaborn as sns
# sns = sns.pairplot(Data_Train_X)
plt.show()
sns.heatmap(Data_Train_X, cmap='coolwarm', annot=True)

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
ax = sns.heatmap(Data_Train_X)
