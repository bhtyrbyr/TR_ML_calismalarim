# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:53:28 2022

@author: muham
"""

"""
    Veri seti içerisinde bulunana eksik verileri belirli yöntemlerle yerini doldurmak
    amaçlanmıştır.
    1. Yöntem olarak pandas kütüphanesinde bulunan `.fillna()` metotu
    2. Yöntem sklearn kütüphanesinin kullanımı
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

nullDataSet1 = pd.read_csv("../DataSets/bilkav_eksikveri_dataset.csv")
nullDataSet2 = pd.read_csv("../DataSets/bilkav_eksikveri_dataset.csv")

# 1. Yol
print("****************************************\n Yol 1\n****************************************")
print("DataSet")
print(nullDataSet1)

nullDataSet1.fillna(nullDataSet1.mean(), inplace = True)

print("****************************************")
print("Sonuç")
print(nullDataSet1)

#2. Yol
print("****************************************\n Yol 2\n****************************************")
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = nullDataSet2.iloc[:,1:4].values
print("DataSet")
print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print("****************************************")
print("Sonuç")
print(yas)