# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:38:29 2022

@author: bhtyr
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy='mean')

dataSet = pd.read_csv("DataSets/dataset.csv")

print(dataSet)