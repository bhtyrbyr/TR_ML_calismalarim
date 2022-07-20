# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""
#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#2 veri önişleme
#2.1 veri yükleme
veriSeti = pd.read_csv('../DataSets/bilkav_tahmin_dataset.csv')

aylar = veriSeti[['Aylar']]
satis = veriSeti[['Satislar']]

#2.6 verilerin eğitim ve test için bölünmesi
#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satis, test_size = 0.33, random_state = 0)
'''
#2.7 verilerin ölçeklenmesi
#from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

'''
#3 model inşası (linear regression)
#from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()


x_test = x_test.sort_index()


plt.title("Aylara Göre Satış")
plt.ylabel("Satışlar")
plt.xlabel("Aylar")
plt.plot(x_train.iloc[:,0], y_train.iloc[:,0], linewidth=1.0)
plt.plot(x_test.iloc[:,0], lr.predict(x_test), linewidth=1.0)


