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
import statsmodels.api as sm
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
    print(type(sonuc), "\n", sonuc)
    print(type(y_test), "\n", y_test)
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
        
############################ Geri Eleme Backward Elimination #########################

"""
Cinsiyeti OLS Regression Result Sonuçları:
    
1. :
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               cinsiyet   R-squared:                       0.626
Model:                            OLS   Adj. R-squared:                  0.509
Method:                 Least Squares   F-statistic:                     5.361
Date:                Thu, 21 Jul 2022   Prob (F-statistic):            0.00440
Time:                        20:35:47   Log-Likelihood:                -5.1425
No. Observations:                  22   AIC:                             22.29
Df Residuals:                      16   BIC:                             28.83
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             2.2338      1.174      1.903      0.075      -0.254       4.722
x2             2.2461      1.075      2.089      0.053      -0.033       4.525
x3             1.8514      1.122      1.651      0.118      -0.526       4.229
x4            -0.0204      0.010     -2.098      0.052      -0.041       0.000
x5             0.0308      0.008      3.682      0.002       0.013       0.048
x6            -0.0077      0.010     -0.813      0.428      -0.028       0.012 (ELENDİ)
==============================================================================
Omnibus:                        0.140   Durbin-Watson:                   1.516
Prob(Omnibus):                  0.932   Jarque-Bera (JB):                0.212
Skew:                          -0.158   Prob(JB):                        0.899
Kurtosis:                       2.637   Cond. No.                     4.53e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.53e+03. This might indicate that there are
strong multicollinearity or other numerical problems.

2. :
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               cinsiyet   R-squared:                       0.611
Model:                            OLS   Adj. R-squared:                  0.519
Method:                 Least Squares   F-statistic:                     6.669
Date:                Thu, 21 Jul 2022   Prob (F-statistic):            0.00204
Time:                        20:35:47   Log-Likelihood:                -5.5883
No. Observations:                  22   AIC:                             21.18
Df Residuals:                      17   BIC:                             26.63
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             2.2654      1.161      1.951      0.068      -0.185       4.715
x2             2.3533      1.056      2.228      0.040       0.125       4.582
x3             1.8116      1.109      1.633      0.121      -0.529       4.152  (ELENDİ)
x4            -0.0220      0.009     -2.347      0.031      -0.042      -0.002
x5             0.0309      0.008      3.737      0.002       0.013       0.048
==============================================================================
Omnibus:                        0.246   Durbin-Watson:                   1.455
Prob(Omnibus):                  0.884   Jarque-Bera (JB):                0.413
Skew:                          -0.184   Prob(JB):                        0.813
Kurtosis:                       2.439   Cond. No.                     4.47e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 4.47e+03. This might indicate that there are
strong multicollinearity or other numerical problems.    

Sonuç : 
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:               cinsiyet   R-squared (uncentered):                   0.775
Model:                            OLS   Adj. R-squared (uncentered):              0.725
Method:                 Least Squares   F-statistic:                              15.49
Date:                Thu, 21 Jul 2022   Prob (F-statistic):                    1.19e-05
Time:                        20:35:47   Log-Likelihood:                         -7.1911
No. Observations:                  22   AIC:                                      22.38
Df Residuals:                      18   BIC:                                      26.75
Df Model:                           4                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.3980      0.211      1.887      0.075      -0.045       0.841
x2             0.6546      0.192      3.417      0.003       0.252       1.057
x3            -0.0072      0.002     -2.924      0.009      -0.012      -0.002
x4             0.0205      0.005      3.734      0.002       0.009       0.032
==============================================================================
Omnibus:                        0.170   Durbin-Watson:                   1.467
Prob(Omnibus):                  0.919   Jarque-Bera (JB):                0.382
Skew:                           0.046   Prob(JB):                        0.826
Kurtosis:                       2.361   Cond. No.                         562.
==============================================================================

Notes:
[1] R² is computed without centering (uncentered) since the model does not contain a constant.
[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

#1:
OLS_cinsiyet = ulkeBoyKiloYasDataFrame.iloc[:,[0,1,2,3,4,5]].values
OLS_cinsiyet = np.array(OLS_cinsiyet,dtype=float)
model = sm.OLS(cinsiyetDataFrame,OLS_cinsiyet).fit()
#(model.summary())
#2:
OLS_cinsiyet = ulkeBoyKiloYasDataFrame.iloc[:,[0,1,2,3,4]].values
OLS_cinsiyet = np.array(OLS_cinsiyet,dtype=float)
model = sm.OLS(cinsiyetDataFrame,OLS_cinsiyet).fit()
#print(model.summary())
#3:
OLS_cinsiyet = ulkeBoyKiloYasDataFrame.iloc[:,[0,1,3,4]].values
OLS_cinsiyet = np.array(OLS_cinsiyet,dtype=float)
model = sm.OLS(cinsiyetDataFrame,OLS_cinsiyet).fit()
#print(model.summary())

# Alınacak Sütünlar: 0, 1, 3, 4

cinsiyetGeriElemeDataFrame = ulkeBoyKiloYasDataFrame.iloc[:,[0,1,3,4]]
#MLRtest(ulkeBoyKiloYasDataFrame, cinsiyetDataFrame, istatistik=1)
#MLRtest(cinsiyetGeriElemeDataFrame, cinsiyetDataFrame, istatistik=1)


"""
"""

#1:
OLS_boy = ulkeKiloYasCinsiyetDF.iloc[:,[0,1,2,3,4,5]].values
OLS_boy = np.array(OLS_boy,dtype=float)
model = sm.OLS(boyDataFrame,OLS_boy).fit()
#print(model.summary())

#1:
OLS_boy = ulkeKiloYasCinsiyetDF.iloc[:,[0,1,2,3,5]].values
OLS_boy = np.array(OLS_boy,dtype=float)
model = sm.OLS(boyDataFrame,OLS_boy).fit()
#print(model.summary())
print("************************************************")
boyGeriElemeDataFrame = ulkeKiloYasCinsiyetDF.iloc[:,[0,1,2,3,5]]
#boy_y_pred = MLRtest(ulkeKiloYasCinsiyetDF, boyDataFrame, standardScaler= False, istatistik = 2)
boy_y_pred = MLRtest(boyGeriElemeDataFrame, boyDataFrame, standardScaler= False, istatistik = 2)

#2:
OLS_boy = ulkeKiloYasCinsiyetDF.iloc[:,[0,1,2,3]].values
OLS_boy = np.array(OLS_boy,dtype=float)
model = sm.OLS(boyDataFrame,OLS_boy).fit()
#print(model.summary())

print("************************************************")
boyGeriElemeDataFrame = ulkeKiloYasCinsiyetDF.iloc[:,[0,1,2,3]]
#boy_y_pred = MLRtest(ulkeKiloYasCinsiyetDF, boyDataFrame, standardScaler= False, istatistik = 2)
boy_y_pred = MLRtest(boyGeriElemeDataFrame, boyDataFrame, standardScaler= False, istatistik = 2)



