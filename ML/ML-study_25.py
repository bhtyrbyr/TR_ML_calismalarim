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
"""
kmeans = KMeans(n_clusters= 3, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans2 = KMeans(n_clusters= i, init = 'k-means++', random_state=123)
    kmeans2.fit(X)
    sonuclar.append(kmeans2.inertia_)
    
plt.plot(range(1,11), sonuclar)
plt.show()
"""
kmeans2 = KMeans(n_clusters = 4, init = 'k-means++', random_state=123)
y_pred1 = kmeans2.fit_predict(X)
plt.scatter(X[y_pred1==0,0],X[y_pred1==0,1], s=100, color='red')
plt.scatter(X[y_pred1==1,0],X[y_pred1==1,1], s=100, color='blue')
plt.scatter(X[y_pred1==2,0],X[y_pred1==2,1], s=100, color='green')
plt.scatter(X[y_pred1==3,0],X[y_pred1==3,1], s=100, color='yellow')
plt.title('KMeans')
plt.show()

#HC
from sklearn.cluster import AgglomerativeClustering

AC = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

y_pred2 = AC.fit_predict(X)
print(y_pred2)

plt.scatter(X[y_pred2==0,0],X[y_pred2==0,1], s=100, color='red')
plt.scatter(X[y_pred2==1,0],X[y_pred2==1,1], s=100, color='blue')
plt.scatter(X[y_pred2==2,0],X[y_pred2==2,1], s=100, color='green')
plt.title('AC')
plt.show()

