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
'''
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
'''
#UCB

import math

N=10000
d=10
oduller = [0] * d
tiklamalar = [0] * d
toplam = 0
secilenler = []


for n in range(1,N):
    add = 0
    max_ucb = 0
    for i in range(0,d):
        if (tiklamalar[i] > 0):
            ortalama = oduller[i] / tiklamalar[i]
            delta = math.sqrt(3/2 * math.log(n)/tiklamalar[i])
            ucb = ortalama + delta
        else:
            ucb = N * 10
        if max_ucb < ucb:
            max_ucb = ucb
            add = i
    secilenler.append(add)
    tiklamalar[add] = tiklamalar[add] + 1
    odul = veriSeti.values[n,add]
    oduller[add] = oduller[add] + odul
    toplam = toplam + odul

print('toplam ödül : ', toplam)


plt.hist(secilenler)
plt.show()
