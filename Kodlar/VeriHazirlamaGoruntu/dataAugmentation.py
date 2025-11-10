#%% Gerekli Kütüphaneler

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#%% Verileri yükleme

train = pd.read_csv("data/train.csv")

print("Egitim verisinin boyutu:" , train.shape)

test = pd.read_csv("data/test.csv")
print("Egitim verisinin boyutu:" , test.shape)

train.head()

Y_train = train["label"]

train.drop(labels = "label", axis =1, inplace = True)

#%% veri analizi

plt.figure(figsize=(15,7))
g = sns.countplot(x=Y_train, palette="icefire")
plt.title("Rakam Sınıflarının SAyısı", fontsize=18)
plt.xlabel("Rakam Sınıfı", fontsize=14)
plt.ylabel("Örnek Sayısı", fontsize=10)
plt.show()

print(Y_train.value_counts().sort_index())

#%% örnek veri görselleri

img = train.iloc[0].values
img = img.reshape((28, 28))
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.title("Örnek Görsel -1", fontsize=18)
plt.axis("off") #eksen çizgisini kaldirir
plt.show()

img = train.iloc[589].values
img = img.reshape((28, 28))
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.title("Örnek Görsel - 2", fontsize=18)
plt.axis("off") #eksen çizgisini kaldirir
plt.show()

#%% Normalizasyon

train = train / 255.0
test = test / 255.0

print("Egitim veri seti piksel değer aralığı: [{:.3f}, {:.3f}]".format(train.min().min(), train.max().max()))


print("Test veri seti piksel değer aralığı: [{:.3f}, {:.3f}]".format(test.min().min(), test.max().max()))


#%% Yeniden şekillendirme

#keras ve tensorflow yani CNN yapısında (ornek sayisi, yükseklik, genislik, kanal)

#(28,28)
# normalizasyon yaptigimiz icin kanal sayisi : 1
#ornek sayimiz veri seti uzunlugu kadar (-1 tüm veri setini içer demek)

train = train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

train.shape

#%% Sınıfların kodlanması

#[0,1,2,3,4]
# 0: [1 0 0 0 0]
# 1: [0 1 0 0 0]
# 2: [0 0 1 0 0]

#[0,1,2,3,4,5,6,7,8,9]
# 0: [1 0 0 0 0 0 0 0 0 0]
# 1: [0 1 0 0 0 0 0 0 0 0]
# 2: [0 0 1 0 0 0 0 0 0 0]
# 9: [0 0 0 0 0 0 0 0 0 1]

num_clases = 10
Y_train = to_categorical(Y_train, num_classes = num_clases)


#%% egitim ve dogrulama ayirimi

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(train, Y_train, test_size=0.1, random_state=13)



plt.figure(figsize=(7,7))
plt.imshow(X_train[5][:, :, 0], cmap = "gray")
plt.title("Yeniden şekillendirilmiş Örnek GÖrüntü")
plt.axis("off")
plt.show()

Y_train[5]

#%% veri arttırma

# Veri seti küçükse
#Overfitting önlemek için
#model genelleştirmek için 
#sınıf dengesizligini 


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center = False, #Veri setinin ortalamasını yapar
    samplewise_center = False, #Her goruntunun ortalamasını alıyor
    featurewise_std_normalization = False, # Veri setindeki verilerin STD Normalizasyonunu sağlıyor
    samplewise_std_normalization = False, #her görüntünün STD Normalizasyonunu sağlıyor
    zca_whitening = False, #ZCA beyazlatma - boyut azaltma işlemi 
    rotation_range = 5, # +5 ve -5 döndürme
    width_shift_range=0.1, # yatay eksende %10 luk görüntüyü kaydırmak
    height_shift_range=0.1, # dikey eksende %10 luk görüntüyü kaydırmak
    horizontal_flip = False, #yatayda çevirme (kullanılan veri seti için uygun değil)
    vertical_flip = False # dikey çevirme
    )

datagen.fit(X_train)


ormek_img = X_train[734]

ormek_img = ormek_img.reshape(1,28,28,1)

#flow metodu veri akışını sağlıyor üretim için kullanılıyor

artirilmis_goruntuler = datagen.flow(ormek_img, batch_size=1)

##orjinal ve arttirilmis goruntuleri ekrana basma

fig, axes = plt.subplots(2, 5, figsize= (20, 8)) #2 satir 5 sütun
fig.suptitle("Veri Arttırma Ornekleri", fontsize=14, fontweight = "bold")

axes[0, 0].imshow(ormek_img[0, :, :, 0], cmap = "gray") #orijinal goruntu
axes[0, 0].set_title("Orijinal Goruntu", fontsize=10)
axes[0, 0].axis("off")

for i in range(1,5):
    axes[0, i].axis("off")
    
for i in range(5):
    uret_gor = next(artirilmis_goruntuler)[0] # bir sonraki üretilmis görüntüyü oluşturur
    axes[1,i].imshow(uret_gor[:, :, 0], cmap= "gray")
    axes[1, i].set_title(f"Uretilmis {i+1}", fontsize=12)
    axes[1, i].axis("off")

plt.tight_layout() #grafiklerin düzenlenmesi için
plt.show()
























