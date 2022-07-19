# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#kodlar
#veri yükleme

veriSeti = pd.read_csv("../DataSets/bilkav_dataset.csv")

print(veriSeti)
print(veriSeti.iloc[:,1:4])
print(veriSeti[(np.abs(veriSeti["kilo"])<35)])

#veri ön işleme




