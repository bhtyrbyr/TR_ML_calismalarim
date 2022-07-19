# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

veriSeti = pd.read_csv('../DataSets/bilkav_dataset.csv')

print(veriSeti)

ulke = veriSeti.iloc[:,0:1].values
print(ulke)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriSeti.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
