# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""

#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#2 veri önişleme
#2.1 veri yükleme
url = 'https://bilkav.com/satislar.csv'
veriler = pd.read_csv(url)
veriler = veriler.values
X = veriler[:,0:1]
y = veriler[:,1]

bolme = 0.33

from sklearn import model_selection

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, y, train_size = bolme,
    random_state=0)
"""
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)

print(lr.predict(X_test))
"""
import pickle as pc

dosya = '../models/model.kayit'
"""
pc.dump(lr, open(dosya,'wb'))
"""
lr_yuklenen = pc.load(open(dosya,'rb'))
print(lr_yuklenen.predict(np.array([5]).reshape(-1,1)))

