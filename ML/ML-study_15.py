# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""
#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#2 veri önişleme
#2.1 veri yükleme
veriSeti = pd.read_csv('../DataSets/bilkav_maaslar_dataset.csv')

x = veriSeti.iloc[:,1:2]
y = veriSeti.iloc[:,2:]
X = x.values
Y = y.values

# linear regression sonuç

lin_reg = LinearRegression()
lin_reg.fit(x.values, y.values)
 
plt.scatter(x.values, y.values, color = 'red')
plt.plot(x.values, lin_reg.predict(x.values), color= 'green')
plt.show()

# polynomial regression

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
y_pred = lin_reg2.predict(x_poly)

plt.scatter(X,Y, color = 'red')
plt.plot(X, y_pred, color = 'yellow')
plt.show()


poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
y_pred2 = lin_reg2.predict(x_poly)
plt.scatter(X,Y, color = 'red')
plt.plot(X, y_pred2, color = 'yellow')
plt.show()
