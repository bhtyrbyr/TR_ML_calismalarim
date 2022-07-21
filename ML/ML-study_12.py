# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:19 2022

@author: muham

    p-value (Olasılık değeri) :  
        H0: Null Hypothesis: Farksızlık hipotezi, sıfır hipotezi, boş hipotez
        H1: Alternatif Hipotez
        P-Değeri : Olasılık değeri (Genelde 0.05 alınır)
        P-Değeri küçüldükçe H0 hatalı olma ihtimali artar
        P-değeri büyüdükçe H1 hatalı olma ihtimali artar
"""
#1. kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#2 veri setinin içeri aktarımı
veriSeti = pd.read_csv('../DataSets/bilkav_dataset.csv')

#2.1 encoder : Kategorik -> Numeric
ulke = veriSeti.iloc[:,0:1].values
cinsiyet = veriSeti.iloc[:,-2:-1].values
le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()
ulke[:,0] = le.fit_transform(veriSeti.iloc[:,0])
cinsiyet[:,0] = le.fit_transform(veriSeti.iloc[:,-1])
ulke = ohe.fit_transform(ulke).toarray()
cinsiyet = ohe.fit_transform(cinsiyet).toarray()

##########################################################

# 2.2 veri setinin birleştirilmesi
yeniVeriSeti = pd.concat([
    pd.DataFrame(ulke, index=range(22), columns=["fr","tr","us"]),
    pd.DataFrame(veriSeti[["boy","kilo","yas"]], index=range(22), columns=["boy","kilo","yas"]),
    pd.DataFrame(cinsiyet[:,0], index=range(22), columns=["cinsiyet"])
    ], axis = 1)

def istatistikYonet(sonuc, test, istatistik = 0):
    if istatistik == 0:
        return 0
    elif istatistik == 1:
        yuvarlama(sonuc)
    elif istatistik == 2:
        hataHesaplama(sonuc, test)
    else:
        return 0
    
        

def yuvarlama(sonuc):
    print(type(sonuc), "\n", sonuc)
    sonuc = np.round(np.abs(sonuc))
    print(sonuc)

def hataHesaplama(sonuc, y_test):
    Sonuc = pd.Series(sonuc)
    Y_test = pd.Series(y_test)
    hatalar = list()
    for i in range(Sonuc.size):
        hata = ((Y_test[Y_test.index[i]] - Sonuc[i]) / Y_test[Y_test.index[i]]) * 100
        hatalar.append(hata)
    hatalarSeries = pd.Series(hatalar)
    print(hatalarSeries.describe())
    plt.title("Tahmin Hataları ")
    plt.ylabel("Hata Oranı(%)")
    plt.xlabel("Tahmin")
    plt.plot(hatalarSeries)


# 3 MLR ile makine öğrenmesi fonksiyonu
def MLRtest(ogrenilecekSet, bulunacakSet, _test_size = 0.33, 
            _random_state = 0, standardScaler = False, istatistik = 0):
    
    x_train, x_test, y_train, y_test = train_test_split(
        ogrenilecekSet,
        bulunacakSet,
        test_size = _test_size,
        random_state = _random_state
        )
    
    regressor = LinearRegression()
    
    if standardScaler == False:
        regressor.fit(x_train,y_train)
        y_pred      = regressor.predict(x_test)
        istatistikYonet(y_pred, y_test, istatistik)
        return y_pred
    else:
        sc          = StandardScaler()
        X_train     = sc.fit_transform(x_train)
        X_test      = sc.fit_transform(x_test)
        Y_train     = sc.fit_transform(y_train)
        Y_test      = sc.fit_transform(y_test)
        regressor.fit(X_train,Y_train)
        y_pred      = regressor.predict(X_test)
        istatistikYonet(y_pred, Y_test, istatistik)
        print(y_pred)
        return y_pred

"""
Cinsiyeti MLR ile tahmin etme 
    Sonuç :
        Test örnek sayısı       = 8
        Doğru tahmin sayısı     = 6
        Yanlış tahmin sayısı    = 2    
"""
ulkeBoyKiloYasDataFrame = yeniVeriSeti[["fr","tr","us","boy","kilo","yas"]]
cinsiyetDataFrame = yeniVeriSeti[["cinsiyet"]]

cinsiyet_y_pred = MLRtest(ulkeBoyKiloYasDataFrame, cinsiyetDataFrame, standardScaler= False, istatistik = 1)

"""
Boyu MLR ile tahmin etme
    Sonuç :
        Test örnek sayısı       = 8
        std                     = 5.894522
        mean                    = -0.527017
        min                     = -11.138041
        25%                     = -4.696337
        50%                     = 1.684255
        75%                     = 2.674818
        max                     = 7.350537
        
"""

boyDataFrame            = yeniVeriSeti["boy"]
ulkeKiloYasCinsiyetDF   = yeniVeriSeti[["fr","tr","us","kilo","yas","cinsiyet"]]

boy_y_pred = MLRtest(ulkeKiloYasCinsiyetDF, boyDataFrame, standardScaler= False, istatistik = 2)
        
    


