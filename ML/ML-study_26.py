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
veriSeti = pd.read_csv('../DataSets/bilkav_ads_ctr_dataset.csv')

N=10000
d=10
toplam = 0
secilenler = []
for n in range(0,N):
    add = random.randrange(d)
    secilenler.append(add)
    odul = veriSeti.values[n,add]
    toplam = toplam + odul
    
plt.hist(secilenler)
plt.show()