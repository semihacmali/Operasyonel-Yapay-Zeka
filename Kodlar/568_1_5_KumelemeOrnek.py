# -*- coding: utf-8 -*-
"""


@author: SemihAcmali
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rng = np.random.default_rng(13)
#rng.normal(Ortalama deger, Standart Sapma, Kaç tane deger olusturulacak)
X = pd.DataFrame({
  "harcama": np.r_[rng.normal(500,120,80),  rng.normal(1500,200,80), rng.normal(300,60,80)],
  "ziyaret": np.r_[rng.normal(3,1,80),      rng.normal(8,1.5,80),    rng.normal(1,0.5,80)],
  "oran":    np.r_[rng.normal(0.05,0.02,80),rng.normal(0.15,0.04,80),rng.normal(0.02,0.01,80)]
})

X_scaled = StandardScaler().fit_transform(X)
k = 3
km = KMeans(n_clusters=k, n_init=10, random_state=13).fit(X_scaled)
labels = km.labels_
X["segment"] = km.labels_
print("Dağılım:\n", X["segment"].value_counts().sort_index())
print("\nProfil ortalamaları:\n", X.groupby("segment").mean().round(2))


# 2B görselleştirme için PCA
pca = PCA(n_components=2, random_state=0)
X_2d = pca.fit_transform(X_scaled)
centers_2d = pca.transform(km.cluster_centers_)

# Grafik: noktalar (kümelere göre) + küme merkezleri
plt.figure()
for c in range(k):
    plt.scatter(X_2d[labels==c, 0], X_2d[labels==c, 1], label=f"Segment {c}")
plt.scatter(centers_2d[:,0], centers_2d[:,1], marker="X", s=200, label="Merkezler")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.title("KMeans Kümeleme (PCA ile 2B projeksiyon)")
plt.legend()
plt.show()