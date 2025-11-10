# -*- coding: utf-8 -*-
"""


@author: SemihAcmali
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[3],[7],[10],[15],[20],[25],[30]]) # Ucusa kalan gun sayisi
y = np.array([2500,2200,2000,1800,1600,1400,1200]) # bilet fiyatı

model = LinearRegression().fit(X, y)
print("Eğim:", model.coef_[0], "Sabit:", model.intercept_)
# y =  3x + 5 

model.coef_[0] * 13 +  model.intercept_
print("25 gün kala tahmin:", model.predict([[25]])[0])


# Tahmin eğrisi için düzgün x aralığı
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

# Grafik: veri noktaları + regresyon doğrusu
plt.figure()
plt.scatter(X, y, label="Veri noktaları")
plt.plot(x_line, y_line, label="Regresyon doğrusu")
plt.xlabel("Uçuğa kalan gün")
plt.ylabel("Fiyat")
plt.title("Basit Doğrusal Regresyon")
plt.legend()
plt.show()