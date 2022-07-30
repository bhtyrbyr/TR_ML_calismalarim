# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""

#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

#2 veri önişleme
#2.1 veri yükleme
veriler = pd.read_csv('../DataSets/bilkav_wine_dataset.csv')


X = veriler.iloc[:,0:13].values
y = veriler.iloc[:,13].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier2 = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
classifier2.fit(X_train2, y_train)

y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm2 = confusion_matrix(y_test, y_pred2)
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm)
print(cm2)
print(cm3)