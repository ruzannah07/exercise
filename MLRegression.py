# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as  pd
dataset = pd.read_csv('gdeltRegion2.csv') 
dataset.iloc[:,8] = [ str(x) for  x  in dataset.iloc[:,8]]

X = dataset.iloc[:,[2,7,8]].values
y = dataset.iloc[:,9].values

## CATEGORICAL DATA 
#===================================================================================================== 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X1 = LabelEncoder()
X[:,0] = labelencoder_X1.fit_transform(X[:,0])
print(labelencoder_X1.classes_)

labelencoder_X2 = LabelEncoder()
X[:,1] = labelencoder_X2.fit_transform(X[:,1])
print(labelencoder_X2.classes_)

labelencoder_X3 = LabelEncoder()
X[:,2] = labelencoder_X3.fit_transform(X[:,2])
print(labelencoder_X3.classes_)

#ONE HOT ENCODER
onehotencoder_X = OneHotEncoder(categorical_features = [0])
onehotencoder_X1 = OneHotEncoder(categorical_features = [1])
onehotencoder_X2 = OneHotEncoder(categorical_features = [2])

X = onehotencoder_X.fit_transform(X).toarray()
X = onehotencoder_X1.fit_transform(X).toarray()
X = onehotencoder_X2.fit_transform(X).toarray()
#=======================================================================================================
"""
ALASAN MELAKUKAN TRAIN TEST SEBELUM STTANNDARISASI.

normalisasi dilakukan dengan menghitung jarak tiap nilai dengan nilai rata2 dari seluruh data. ketika dilakukan
normalisasi terlebihdahulu sebelum  train test split,maka  nilai rata rata data pada masing2 traintest adalah nilai rata2
keseluruhan. alias, gabungan dari train test split.sedangkan TRAIN TEST SPLIT DI PROSES SECARA TERPISAH
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25 , random_state=0)
#=======================================================================================================
"""
tidak diperlukan untuk normalisasi karna datanya berbentuk kategorikal
from sklearn.preprocessing import StandarScaler

sc_X = StandarScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# CARI NILAI S SQUARED
#lihat var mana yang paling berpengaruh dalam penentuann avggTone
#=======================================================================================================
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
#random state = 0. because we want to get the same result
regressor.fit(X, y)#fitting our regressor obj to our dataset
#X is our matrix of features
#y is our dependent variable vector

# Predicting a new result
y_predDT = regressor.predict(X_test)
# visualisasi 3//4 dimensi
#=======================================================================================================
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#fit the regressor object to our TRAINING SET
regressor.fit(X_train, y_train)

y_predLinReg = regressor.predict(X_test)
regressor.intercept_
#=======================================================================================================
import numpy as np
def CostFunc(y_pred,y_test):    
    m = len(y_test)
    for x in range(0,len(y_pred)):
        s = np.power((y_pred[x] - y_test[x]), 2)
    J = (1.0/(2*m))*s.sum(axis=0)
    return J    

CostFunc(y_pred,y_test)