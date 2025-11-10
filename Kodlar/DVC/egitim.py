# -*- coding: utf-8 -*-
"""
@author: SemihAcmali
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

#Veriyi alma
data = pd.read_csv("deneyim_maas.csv")

X = data[['deneyim_yili']]
y = data['maas']


#Model Eğitimi

model = LinearRegression()
model.fit(X, y)


#model kaydetme
import joblib

joblib.dump(model, "model.pkl")
