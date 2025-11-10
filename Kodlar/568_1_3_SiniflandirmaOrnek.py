# -*- coding: utf-8 -*-
"""

@author: SemihAcmali
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Rastgele veri seti oluşturma
X, y = make_classification(n_samples=100, # 100 tane rastgele deger olusturacak
                           n_features=2,   # görselleştirme için 2 özellik
                           n_classes=2,  # etiket sayısı 2 olacak
                           n_informative=2, #anlamlı özellik sayısı
                           n_redundant=0,  #anlamsız özellik sayısı
                           random_state=13) 

# 2. Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# 3. Modeli oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Başarı oranı
accuracy = model.score(X_test, y_test)
print("Doğruluk Oranı:", accuracy)

# 5. Karar sınırını çizmek için grid oluşturma
import numpy as np
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Tahmin olasılıklarını al
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 6. Grafik çizimi
plt.contourf(xx, yy, Z, alpha=0.3)  # karar sınırları
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')  # veri noktaları
plt.title("Lojistik Regresyon Karar Sınırı")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.show()
