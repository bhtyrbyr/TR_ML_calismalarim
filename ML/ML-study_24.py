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
veriSeti = pd.read_csv('../DataSets/bilkav_musteriler_dataset.csv')

#2.5 dataframe yapılarını tek bir dataframe yapısı altında toparlam

X = veriSeti.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters= 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters= i, init = 'k-means++', random_state=123)
    kmeans2.fit(X)
    print('---------------------------------------')
    sonuclar.append(kmeans2.inertia_)
    
plt.plot(range(1,11), sonuclar)