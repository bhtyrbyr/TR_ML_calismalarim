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
lin_reg.fit(X, Y)
 
plt.scatter(x.values, y.values, color = 'red')
plt.plot(X, lin_reg.predict(X), color= 'green')
plt.show()

# polynomial regression

poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
y_pred = lin_reg2.predict(x_poly)

plt.scatter(X,Y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.show()

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
y_pred2 = lin_reg2.predict(x_poly)
plt.scatter(X,Y, color = 'red')
plt.plot(X, y_pred2, color = 'magenta')
plt.show()

print(lin_reg.predict(np.array([11]).reshape(-1,1)))
print(lin_reg.predict(np.array([6.6]).reshape(-1,1)))

print(lin_reg2.predict(poly_reg.fit_transform(np.array([11]).reshape(-1,1))))
print(lin_reg2.predict(poly_reg.fit_transform(np.array([6.6]).reshape(-1,1))))

sc1 = StandardScaler()
sc2 = StandardScaler()

x_olcekli = sc1.fit_transform(X)
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR

sv_reg = SVR(kernel='linear')
sv_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli, color='red')
plt.plot(x_olcekli, sv_reg.predict(x_olcekli), color= 'blue')

print(sv_reg.predict(np.array([11]).reshape(-1,1)))
print(sv_reg.predict(np.array([6.6]).reshape(-1,1)))

















