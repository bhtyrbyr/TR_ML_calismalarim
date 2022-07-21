# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham

    p-value (Olasılık değeri) :  
        H0: Null Hypothesis: Farksızlık hipotezi, sıfır hipotezi, boş hipotez
        H1: Alternatif Hipotez
        P-Değeri : Olasılık değeri (Genelde 0.05 alınır)
        P-Değeri küçüldükçe H0 hatalı olma ihtimali artar
        P-değeri büyüdükçe H1 hatalı olma ihtimali artar
"""
#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#2 veri setinin içeri aktarımı
dataSet = pd.read_csv('../DataSets/bilkav_odev_dataset.csv')
print(dataSet)
#2.1 encoder : Kategorik -> Numeric

encodeDataSet = dataSet.apply(preprocessing.LabelEncoder().fit_transform)
ohe = preprocessing.OneHotEncoder()
outlookEncode = ohe.fit_transform(encodeDataSet.iloc[:,:1]).toarray()
"""
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

outlook = veriSeti.loc[:,["outlook"]].values
windy = veriSeti.loc[:,["windy"]].values
play = veriSeti.loc[:,["play"]].values

outlook[:,0] = le.fit_transform(veriSeti.iloc[:,0])
windy[:,0] = le.fit_transform(veriSeti.iloc[:,3])
play[:,0] = le.fit_transform(veriSeti.iloc[:,4])

outlook = ohe.fit_transform(outlook).toarray()
windy = ohe.fit_transform(windy).toarray()
play = ohe.fit_transform(play).toarray()
"""
##########################################################

# 2.2 veri setinin birleştirilmesi
newDataSet = pd.concat([
    pd.DataFrame(outlookEncode,  index=range(len(dataSet.index)), 
                 columns=["overcast","rainy","sunny"]),
    
    pd.DataFrame(dataSet[["temperature","humidity"]],  
                 index=range(len(dataSet.index)), 
                 columns=["temperature","humidity"]),
    
    pd.DataFrame(encodeDataSet[["windy", "play"]],  
                 index=range(len(dataSet.index)), 
                 columns=["windy", "play"]),
    ], axis=1)
"""
yeniVeriSeti = pd.concat([
    pd.DataFrame(outlook, index=range(len(veriSeti.index)), 
                 columns=["overcast","rainy","sunny"]),
    
    pd.DataFrame(veriSeti[["temperature","humidity"]], 
                 index=range(len(veriSeti.index)), 
                 columns=["temperature","humidity"]),
    
    pd.DataFrame(windy[:,-1:], index=range(len(veriSeti.index)), 
                 columns=["windy"]),
    
    pd.DataFrame(play[:,-1:], index=range(len(veriSeti.index)), 
                 columns=["play"]),
    ], axis = 1)
"""
def istatistikYonet(y_pred, y_test, statistics = 0):
    if statistics == 0:
        return 0
    elif statistics == 1:
        _round(y_pred, y_test)
    elif statistics == 2:
        calculateError(y_pred, y_test)
    else:
        return 0
    
        

def _round(y_pred, y_test):
    print(type(y_pred), "\n", y_pred)
    print(type(y_test), "\n", y_test)
    y_pred = np.round(np.abs(y_pred))
    print(y_pred)

def calculateError(y_pred, y_test):
    print("------------\nY_pred:\n",type(y_pred), "\n", y_pred)
    print("------------\nY_test:\n",type(y_test), "\n", y_test)
    y_pred = pd.Series(y_pred[:,0])
    Y_test = pd.Series(y_test.iloc[:,-1])
    print(type(y_pred), "\n", y_pred)
    print(type(y_test), "\n", y_test)
    errs = list()
    for i in range(y_pred.size): 
        err = ((Y_test[Y_test.index[i]] - y_pred[i]) / Y_test[Y_test.index[i]]) * 100
        errs.append(err)
    errsSeries = pd.Series(errs)
    print(errsSeries)
    print(errsSeries.describe())
    plt.title("Forecast Errors ")
    plt.ylabel("Err Rate(%)")
    plt.xlabel("Forecast")
    plt.plot(errsSeries)


# 3 MLR ile makine öğrenmesi fonksiyonu
def MLRtest(independentDataSet, dependentDataSet, _test_size = 0.33, 
            _random_state = 0, standardScaler = False, statistics = 0):
    
    x_train, x_test, y_train, y_test = train_test_split(
        independentDataSet,
        dependentDataSet,
        test_size = _test_size,
        random_state = _random_state
        )
    regressor = LinearRegression()
    print(
        type(x_train),"\n",
        type(x_test),"\n",
        type(y_train),"\n",
        type(y_test),"\n",
        )
    if standardScaler == False:
        regressor.fit(x_train,y_train)
        y_pred      = regressor.predict(x_test)
        istatistikYonet(y_pred, y_test, statistics)
        return y_pred
    else:
        sc          = StandardScaler()
        X_train     = sc.fit_transform(x_train)
        X_test      = sc.fit_transform(x_test)
        Y_train     = sc.fit_transform(y_train)
        Y_test      = sc.fit_transform(y_test)
        regressor.fit(X_train,Y_train)
        y_pred      = regressor.predict(X_test)
        istatistikYonet(y_pred, Y_test, statistics)
        print(y_pred)
        return y_pred

"""
Play'i MLR ile tahmin etme 
    Sonuç :
        Test örnek sayısı       = 5
        Doğru tahmin sayısı     = 4
        Yanlış tahmin sayısı    = 1   
"""
independentDataFrame = newDataSet[["overcast","rainy","sunny","temperature","humidity",
                                     "windy"]]
dependentDataFrame = newDataSet[["play"]]


OLS_independentData = independentDataFrame.iloc[:,[0,1,2,3,4,5]].values
OLS_independentData = np.array(OLS_independentData, dtype=float)
model = sm.OLS(dependentDataFrame, OLS_independentData).fit()
print(model.summary())

OLS_independentData = independentDataFrame.iloc[:,[0,1,2,5]].values
OLS_independentData = np.array(OLS_independentData, dtype=float)
model = sm.OLS(dependentDataFrame, OLS_independentData).fit()
print(model.summary())

OLS_independentData = independentDataFrame.iloc[:,[0,1,2]].values
OLS_independentData = np.array(OLS_independentData, dtype=float)
model = sm.OLS(dependentDataFrame, OLS_independentData).fit()
print(model.summary())

OLS_independentData = pd.DataFrame(OLS_independentData)
play_y_pred = MLRtest(independentDataFrame, dependentDataFrame, standardScaler= False, statistics = 1)
play_y_pred = MLRtest(OLS_independentData, dependentDataFrame, standardScaler= False, statistics = 1)

"""
Humidity'i MLR ile tahmin etme

"""
independentDataFrame = newDataSet[["overcast","rainy","sunny","temperature","windy",
                                     "play"]]
dependentDataFrame = newDataSet[["humidity"]]

OLS_independentData = independentDataFrame.iloc[:,[0,1,2,3,4,5]].values
OLS_independentData = np.array(OLS_independentData, dtype=float)
model = sm.OLS(dependentDataFrame, OLS_independentData).fit()
print(model.summary())

OLS_independentData = independentDataFrame.iloc[:,[0,1,2,3,5]].values
OLS_independentData = np.array(OLS_independentData, dtype=float)
model = sm.OLS(dependentDataFrame, OLS_independentData).fit()
print(model.summary())

OLS_independentData = pd.DataFrame(OLS_independentData)
humidity_y_pred = MLRtest(independentDataFrame, dependentDataFrame, standardScaler= False, statistics = 2)
humidity_y_pred = MLRtest(OLS_independentData, dependentDataFrame, standardScaler= False, statistics = 2)
