# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 23:03:13 2025

@author: SemihAcmali
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
import numpy as np

# Örnek veri (XOR problemi)
X, y = make_classification(n_samples=1000, # 100 tane rastgele deger olusturacak
                           n_features=5,   # görselleştirme için 2 özellik
                           n_classes=2,  # etiket sayısı 2 olacak
                           n_informative=2, #anlamlı özellik sayısı
                           n_redundant=1,  #anlamsız özellik sayısı
                           random_state=13) 

# Ağ yapısı: 2 giriş, 1 gizli katman (3 nöron), 1 çıktı
model = Sequential([
    Dense(3, input_dim=5, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Derleme: optimizer, loss, metrik
model.compile(optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["accuracy"])

# Eğitim
history = model.fit(X, y, epochs=10, verbose=0)

# Değerlendirme (loss + accuracy)
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Tahminler
preds = model.predict(X)
print("Tahminler (yuvarlatılmış):", preds.round().flatten())
print("Gerçek değerler:", y.flatten())


import matplotlib.pyplot as plt

# Eğitim geçmişinden loss ve accuracy değerlerini al
loss = history.history["loss"]
acc = history.history["accuracy"]

epochs = range(1, len(loss)+1)

plt.figure(figsize=(10,4))

# Loss grafiği
plt.subplot(1,2,1)
plt.plot(epochs, loss, "r-", label="Eğitim kaybı (loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Süreci - Loss")
plt.legend()

# Accuracy grafiği
plt.subplot(1,2,2)
plt.plot(epochs, acc, "b-", label="Eğitim doğruluğu (accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Eğitim Süreci - Accuracy")
plt.legend()

plt.tight_layout()
plt.show()