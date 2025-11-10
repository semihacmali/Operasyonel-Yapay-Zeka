# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 17:58:06 2025

@author: SemihAcmali
"""

# Gerekli kütüphaneleri içe aktarma
import numpy as np # Sayısal işlemler için numpy kütüphanesi
import pandas as pd # Veri işleme ve CSV dosya okuma için pandas kütüphanesi

import matplotlib.pyplot as plt # Görselleştirme için matplotlib kütüphanesi

import seaborn as sns # İleri düzey görselleştirme için seaborn kütüphanesi

from collections import Counter # Sayma işlemleri için Counter sınıfı



# =============================================================================
# VERİ YÜKLEME
# =============================================================================

# Eğitim veri setini yükleme
train_df = pd.read_csv("data/train.csv")
# Test veri setini yükleme
test_df = pd.read_csv("data/test.csv")
# Test veri setindeki PassengerId'leri saklama (daha sonra kullanmak için)
test_PassengerId = test_df["PassengerId"]

# Veri setinin sütun isimlerini görüntüleme
train_df.columns

# Veri setinin ilk 5 satırını görüntüleme
train_df.head()

# Veri setinin sayısal değişkenler için istatistiksel özetini görüntüleme
train_df.describe()

# =============================================================================
# DEĞİŞKEN AÇIKLAMALARI
# =============================================================================

#     PassengerId: Her yolcuya özgü benzersiz kimlik numarası
#     Survived: Yolcunun hayatta kalma durumu (1 = hayatta kaldı, 0 = öldü)
#     Pclass: Yolcu sınıfı (1 = birinci sınıf, 2 = ikinci sınıf, 3 = üçüncü sınıf)
#     Name: Yolcunun adı
#     Sex: Yolcunun cinsiyeti
#     Age: Yolcunun yaşı
#     SibSp: Kardeş/eş sayısı
#     Parch: Ebeveyn/çocuk sayısı
#     Ticket: Bilet numarası
#     Fare: Bilet için ödenen ücret miktarı
#     Cabin: Kamara kategorisi
#     Embarked: Yolcunun bindiği liman (C = Cherbourg, Q = Queenstown, S = Southampton)

# Veri setinin genel bilgilerini görüntüleme (sütun sayısı, veri tipleri, eksik değerler)
train_df.info()


# =============================================================================
# TEK DEĞİŞKENLİ ANALİZ (UNIVARIATE VARIABLE ANALYSIS)
# =============================================================================

#     Kategorik Değişkenler: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, SibSp ve Parch
#     Sayısal Değişkenler: Fare, Age ve PassengerId

# Kategorik değişkenler için bar grafik çizme fonksiyonu
def bar_plot(variable):
    """
        Girdi: değişken adı örn: "Sex"
        Çıktı: bar grafik ve değer sayıları
    """
    # Değişkeni al
    var = train_df[variable]
    # Kategorik değişkenin değer sayılarını hesapla
    varValue = var.value_counts()
    
    # Görselleştir
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frekans")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    
    
# Birinci kategori değişkenler için bar grafik çizme
category1 = ["Survived","Sex","Pclass","Embarked","SibSp", "Parch"]
for c in category1:
    bar_plot(c)
    

# İkinci kategori değişkenler için sadece değer sayılarını yazdırma (çok fazla kategori olduğu için)
category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
    
# Sayısal değişkenler için histogram çizme fonksiyonu
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frekans")
    plt.title("{} dağılımı histogram ile".format(variable))
    plt.show()
    


# Sayısal değişkenler için histogram çizme
numericVar = ["Fare", "Age","PassengerId"]
for n in numericVar:
    plot_hist(n)

# =============================================================================
# İKİ DEĞİŞKENLİ ANALİZ (BIVARIATE VARIABLE ANALYSIS)
# =============================================================================

# Sınıf (Pclass) ile hayatta kalma (Survived) arasındaki ilişkiyi analiz etme
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# Cinsiyet (Sex) ile hayatta kalma (Survived) arasındaki ilişkiyi analiz etme
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# Kardeş/Eş sayısı (SibSp) ile hayatta kalma (Survived) arasındaki ilişkiyi analiz etme
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean().sort_values(by="Survived",ascending = False)


# Ebeveyn/Çocuk sayısı (Parch) ile hayatta kalma (Survived) arasındaki ilişkiyi analiz etme
train_df[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by="Survived",ascending = False)

# =============================================================================
# AYKIRI DEĞER TESPİTİ (OUTLIER DETECTION)
# =============================================================================

# IQR (Interquartile Range) yöntemi kullanarak aykırı değerleri tespit eden fonksiyon
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1. çeyrek (Q1 - %25'lik değer)
        Q1 = np.percentile(df[c],25)
        # 3. çeyrek (Q3 - %75'lik değer)
        Q3 = np.percentile(df[c],75)
        # IQR (Interquartile Range) hesaplama
        IQR = Q3 - Q1
        # Aykırı değer adımı (1.5 * IQR)
        outlier_step = IQR * 1.5
        # Aykırı değerleri ve indekslerini tespit etme
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # İndeksleri saklama
        outlier_indices.extend(outlier_list_col)
    
    # Birden fazla özellikte aykırı değer olan kayıtları bulma
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

# Tespit edilen aykırı değerleri görüntüleme
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]

# Aykırı değerleri veri setinden çıkarma
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

# =============================================================================
# EKSİK DEĞER İŞLEME (MISSING VALUE HANDLING)
# =============================================================================

#     Eksik Değerleri Bulma
#     Eksik Değerleri Doldurma

# Eğitim veri setinin uzunluğunu saklama (daha sonra ayırmak için)
train_df_len = len(train_df)
# Eğitim ve test veri setlerini birleştirme (eksik değer doldurma için)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)

train_df.head()

# Eksik değer içeren sütunları bulma
train_df.columns[train_df.isnull().any()]

# Her sütundaki eksik değer sayısını görüntüleme
train_df.isnull().sum()

# Embarked sütununda eksik değer olan kayıtları görüntüleme
train_df[train_df["Embarked"].isnull()]

# Embarked'a göre Fare değişkeninin boxplot grafiğini çizme (eksik değer doldurma stratejisi belirlemek için)
train_df.boxplot(column="Fare",by = "Embarked")
plt.show()

# Embarked eksik değerlerini "C" (Cherbourg) ile doldurma
train_df["Embarked"] = train_df["Embarked"].fillna("C")
# Doldurma işleminin başarılı olduğunu kontrol etme
train_df[train_df["Embarked"].isnull()]

# Fare sütununda eksik değer olan kayıtları görüntüleme
train_df[train_df["Fare"].isnull()]

# Fare eksik değerlerini 3. sınıf yolcuların ortalama ücreti ile doldurma
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"] == 3]["Fare"]))

# Doldurma işleminin başarılı olduğunu kontrol etme
train_df[train_df["Fare"].isnull()]

# =============================================================================
# GÖRSELLEŞTİRME (VISUALIZATION)
# =============================================================================

# SibSp, Parch, Age, Fare ve Survived arasındaki korelasyonları görselleştirme

# Kardeş/Eş sayısı (SibSp) ile hayatta kalma olasılığı arasındaki ilişkiyi görselleştirme
g = sns.factorplot(x = "SibSp", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Hayatta Kalma Olasılığı")
plt.show()


# Ebeveyn/Çocuk sayısı (Parch) ile hayatta kalma olasılığı arasındaki ilişkiyi görselleştirme
g = sns.factorplot(x = "Parch", y = "Survived", kind = "bar", data = train_df, size = 6)
g.set_ylabels("Hayatta Kalma Olasılığı")
plt.show()


# Sınıf (Pclass) ile hayatta kalma olasılığı arasındaki ilişkiyi görselleştirme
g = sns.factorplot(x = "Pclass", y = "Survived", data = train_df, kind = "bar", size = 6)
g.set_ylabels("Hayatta Kalma Olasılığı")
plt.show()

# Hayatta kalma durumuna göre yaş dağılımını görselleştirme
g = sns.FacetGrid(train_df, col = "Survived")
g.map(sns.distplot, "Age", bins = 25)
plt.show()

# Hayatta kalma durumu ve sınıfa göre yaş dağılımını görselleştirme
g = sns.FacetGrid(train_df, col = "Survived", row = "Pclass", size = 2)
g.map(plt.hist, "Age", bins = 25)
g.add_legend()
plt.show()

# Biniş limanına göre sınıf, hayatta kalma ve cinsiyet ilişkisini görselleştirme
g = sns.FacetGrid(train_df, row = "Embarked", size = 2)
g.map(sns.pointplot, "Pclass","Survived","Sex")
g.add_legend()
plt.show()


# Biniş limanı ve hayatta kalma durumuna göre cinsiyet ve ücret ilişkisini görselleştirme
g = sns.FacetGrid(train_df, row = "Embarked", col = "Survived", size = 2.3)
g.map(sns.barplot, "Sex", "Fare")
g.add_legend()
plt.show()


# =============================================================================
# EKSİK DEĞER DOLDURMA: YAŞ ÖZELLİĞİ (FILL MISSING: AGE FEATURE)
# =============================================================================

# Yaş sütununda eksik değer olan kayıtları görüntüleme
train_df[train_df["Age"].isnull()]

# Cinsiyete göre yaş dağılımını boxplot ile görselleştirme
sns.factorplot(x = "Sex", y = "Age", data = train_df, kind = "box")
plt.show()

# Cinsiyet ve sınıfa göre yaş dağılımını boxplot ile görselleştirme
sns.factorplot(x = "Sex", y = "Age", hue = "Pclass",data = train_df, kind = "box")
plt.show()

# Ebeveyn/Çocuk sayısı ve Kardeş/Eş sayısına göre yaş dağılımını görselleştirme
sns.factorplot(x = "Parch", y = "Age", data = train_df, kind = "box")
sns.factorplot(x = "SibSp", y = "Age", data = train_df, kind = "box")
plt.show()

# Yaş ile diğer özellikler arasındaki korelasyonu heatmap ile görselleştirme
sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True)
plt.show()

# Yaş eksik değerlerini doldurma: SibSp, Parch ve Pclass'a göre medyan yaş kullanma
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    # Aynı SibSp, Parch ve Pclass değerlerine sahip kayıtların medyan yaşını hesaplama
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    # Genel medyan yaş
    age_med = train_df["Age"].median()
    # Eğer tahmin edilen yaş geçerliyse onu kullan, değilse genel medyanı kullan
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med

# Doldurma işleminin başarılı olduğunu kontrol etme
train_df[train_df["Age"].isnull()]


# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ: İSİM -- UNVAN (FEATURE ENGINEERING: NAME -- TITLE)
# =============================================================================

# İsim sütununun ilk 10 değerini görüntüleme
train_df["Name"].head(10)

# İsim sütunundan unvan (Title) çıkarma
name = train_df["Name"]
# İsimden unvanı ayıklama (örn: "Braund, Mr. Owen Harris" -> "Mr")
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]

train_df["Title"].head(10)

# Unvanların dağılımını görselleştirme
sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()

# Unvanları kategorik değerlere dönüştürme
# Nadir görülen unvanları "other" kategorisine birleştirme
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
# Unvanları sayısal değerlere dönüştürme: Master=0, Miss/Ms/Mlle/Mrs=1, Mr=2, Other=3
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(20)

# Dönüştürülmüş unvanların dağılımını görselleştirme
sns.countplot(x="Title", data = train_df)
plt.xticks(rotation = 60)
plt.show()

# Unvan ile hayatta kalma olasılığı arasındaki ilişkiyi görselleştirme
g = sns.factorplot(x = "Title", y = "Survived", data = train_df, kind = "bar")
g.set_xticklabels(["Master","Mrs","Mr","Other"])
g.set_ylabels("Hayatta Kalma Olasılığı")
plt.show()

# İsim sütununu veri setinden çıkarma (artık unvan özelliği var)
train_df.drop(labels = ["Name"], axis = 1, inplace = True)

train_df.head()

# Unvan özelliğini one-hot encoding ile dönüştürme
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df.head()




# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ: AİLE BÜYÜKLÜĞÜ (FEATURE ENGINEERING: FAMILY SIZE)
# =============================================================================

train_df.head()

# Aile büyüklüğü özelliğini oluşturma (kardeş/eş + ebeveyn/çocuk + kendisi)
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1

train_df.head()

# Aile büyüklüğü ile hayatta kalma olasılığı arasındaki ilişkiyi görselleştirme
g = sns.factorplot(x = "Fsize", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Hayatta Kalma")
plt.show()

# Aile büyüklüğünü kategorik değere dönüştürme (küçük aile: <5, büyük aile: >=5)
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]

train_df.head(10)

# Aile büyüklüğü kategorilerinin dağılımını görselleştirme
sns.countplot(x = "family_size", data = train_df)
plt.show()

# Aile büyüklüğü kategorisi ile hayatta kalma olasılığı arasındaki ilişkiyi görselleştirme
g = sns.factorplot(x = "family_size", y = "Survived", data = train_df, kind = "bar")
g.set_ylabels("Hayatta Kalma")
plt.show()

# Aile büyüklüğü özelliğini one-hot encoding ile dönüştürme
train_df = pd.get_dummies(train_df, columns= ["family_size"])
train_df.head()

# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ: BİNİŞ LİMANI (FEATURE ENGINEERING: EMBARKED)
# =============================================================================

train_df["Embarked"].head()

# Biniş limanlarının dağılımını görselleştirme
sns.countplot(x = "Embarked", data = train_df)
plt.show()

# Biniş limanı özelliğini one-hot encoding ile dönüştürme
train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head()




# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ: BİLET (FEATURE ENGINEERING: TICKET)
# =============================================================================

train_df["Ticket"].head(20)

# Bilet numarasından önek çıkarma örneği
a = "A/5. 2151"
a.replace(".","").replace("/","").strip().split(" ")[0]

# Bilet numaralarından önekleri çıkarma ve temizleme
tickets = []
for i in list(train_df.Ticket):
    # Eğer bilet numarası sadece rakamlardan oluşmuyorsa, önekini çıkar
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        # Sadece rakamlardan oluşuyorsa "x" olarak işaretle
        tickets.append("x")
train_df["Ticket"] = tickets

train_df["Ticket"].head(20)

train_df.head()

# Bilet özelliğini one-hot encoding ile dönüştürme (T_ öneki ile)
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")
train_df.head(10)




# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ: SINIF (FEATURE ENGINEERING: PCLASS)
# =============================================================================

# Sınıf dağılımını görselleştirme
sns.countplot(x = "Pclass", data = train_df)
plt.show()

# Sınıf özelliğini kategorik tipe dönüştürme
train_df["Pclass"] = train_df["Pclass"].astype("category")
# Sınıf özelliğini one-hot encoding ile dönüştürme
train_df = pd.get_dummies(train_df, columns= ["Pclass"])
train_df.head()




# =============================================================================
# ÖZELLİK MÜHENDİSLİĞİ: CİNSİYET (FEATURE ENGINEERING: SEX)
# =============================================================================

# Cinsiyet özelliğini kategorik tipe dönüştürme
train_df["Sex"] = train_df["Sex"].astype("category")
# Cinsiyet özelliğini one-hot encoding ile dönüştürme
train_df = pd.get_dummies(train_df, columns=["Sex"])
train_df.head()

# =============================================================================
# SON TEMİZLİK: PASSENGER ID VE CABIN SÜTUNLARINI ÇIKARMA
# =============================================================================

# PassengerId ve Cabin sütunlarını veri setinden çıkarma (model için gerekli değil)
train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)

# Son durumdaki sütun isimlerini görüntüleme
train_df.columns

