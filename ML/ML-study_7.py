# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""
#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#2 veri önişleme
#2.1 veri yükleme
veriSeti = pd.read_csv('../DataSets/bilkav_eksikveri_dataset.csv')

#2.2 eksik veri işlemi "mean"
#from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
yas = veriSeti.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4]) 

#2.3 encoder : Kategorik -> Numeric
#from sklearn import preprocessing
ulke = veriSeti.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriSeti.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

##2.4 numpy dizilerini dataframe yapısına çevirme işlemi 
#Ulke encoding'in sonucunu dataframe yapısına çeviriyoruz.
sonuc1 = pd.DataFrame(ulke, index=range(22), columns=["fr","tr","us"])
#aynı şekilde yaş işlemi için yapılanı da dataframe yapısına çeviriyoruz.
sonuc2 = pd.DataFrame(yas, index=range(22), columns=["boy","kilo","yas"])
#son olarak ilk veri setinde bulunan cinsiyet verisini çekiyoruz
sonuc3 = pd.DataFrame(veriSeti.iloc[:,-1], index=range(22), columns=["cinsiyet"])

#2.5 dataframe yapılarını tek bir dataframe yapısı altında toparlama
ilkBirlesim = pd.concat([sonuc1,sonuc2], axis = 1)
yeniVeriSeti = pd.concat([ilkBirlesim, sonuc3], axis = 1)
print(yeniVeriSeti)

#Ekstra - Oluşturulan dataframe'i tekrar yazdırma işlemi
#oluşturulan düzenlenmiş dataseti tekrar yazdırıyoruz
yeniVeriSeti.to_csv("../DataSets/bilkav_islenmis_dataset.csv", index=False)

#2.6 verilerin eğitim ve test için bölünmesi
#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ilkBirlesim, sonuc3, test_size = 0.33, random_state = 0)

#2.7 verilerin ölçeklenmesi
#from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)











