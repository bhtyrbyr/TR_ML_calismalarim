# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


veriSeti = pd.read_csv('../DataSets/bilkav_eksikveri_dataset.csv')

print("**********************\n veri seti\n**********************\n" , veriSeti)

#from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = veriSeti.iloc[:,1:4].values
print("**********************\neksik veri doldurulmadan öncesi\n**********************\n",yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4]) 
print("**********************\neksik veri doldurulmadan sonrası\n**********************\n",yas)


#from sklearn import preprocessing

ulke = veriSeti.iloc[:,0:1].values
print("**********************\nülke stunu\n**********************\n",ulke)
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriSeti.iloc[:,0])
print("**********************\nülke stunu encoding işlemi\n**********************\n",ulke)
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print("**********************\nülke stunu encoding işlemi sonucu\n**********************\n",ulke)

#Ulke encoding'in sonucunu dataframe yapısına çeviriyoruz.
sonuc1 = pd.DataFrame(ulke, index=range(22), columns=["fr","tr","us"])
print("**********************\nülke stunu dataframe yapısı\n**********************\n",sonuc1)

#aynı şekilde yaş işlemi için yapılanı da dataframe yapısına çeviriyoruz.
sonuc2 = pd.DataFrame(yas, index=range(22), columns=["boy","kilo","yas"])
print("**********************\nülke stunu dataframe yapısı\n**********************\n",sonuc2)

#son olarak ilk veri setinde bulunan cinsiyet verisini çekiyoruz
sonuc3 = pd.DataFrame(veriSeti.iloc[:,-1], index=range(22), columns=["cinsiyet"])
print("**********************\nülke stunu dataframe yapısı\n**********************\n",sonuc3)


ilkBirlesim = pd.concat([sonuc1,sonuc2], axis = 1)
yeniVeriSeti = pd.concat([ilkBirlesim, sonuc3], axis = 1)
print(yeniVeriSeti)

#oluşturulan düzenlenmiş dataseti tekrar yazdırıyoruz

yeniVeriSeti.to_csv("../DataSets/bilkav_islenmis_dataset.csv", index=False)

#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(ilkBirlesim, sonuc3, test_size = 0.33, random_state = 0)