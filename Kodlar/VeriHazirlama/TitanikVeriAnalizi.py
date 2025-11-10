# -*- coding: utf-8 -*-
"""
@author: SemihAcmali
"""

#%%  Kütüphaneler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

#%% Veri Yükleme

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

raw_train_df = train_df.copy()
raw_test_df = test_df.copy()
#test verisi icinb hazirlik

test_PassengerId = test_df["PassengerId"]

train_df.columns

train_df.head(20)

train_df.info()
train_df.describe()

train_df.drop(labels = "Unnamed: 0", axis = 1, inplace = True)
test_df.drop(labels = "Unnamed: 0", axis = 1, inplace = True)


#%% Kategorik verilerin Analizi


def bar_plot(variable):

    var = train_df[variable]
    varValue = var.value_counts()
    
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frekans")
    plt.title(variable)
    plt.show()
    
    print("{}: \n {}".format(variable,varValue))
 
categorik1 = ["Survived", "Pclass", "SibSp", "Parch", "Embarked", "Gender"]

for c in categorik1:
    bar_plot(c)
    
categorik2 = ["Cabin", "Ticket", "Name"]

for c in categorik2:
    print("{}: \n".format(train_df[c].value_counts()))

def hist_plot(variable):
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frekans")
    plt.title("{} dağılımı".format(variable))
    plt.show()


sayisalDegerler = ["Fare", "Age", "PassengerId"]

for s in sayisalDegerler:
    hist_plot(s)
    
#%% Sql gibi veri analizi

# Sınıf (Pclass) ile Survived arasındaki iliski

train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by = "Survived", ascending = False)

#cinsiyet ile survived arasındaki iliski
    
train_df[["Gender", "Survived"]].groupby(["Gender"], as_index = False).mean().sort_values(by = "Survived", ascending = False)   
    
#Parch ile survived arasındaki iliski
    
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index = False).mean().sort_values(by = "Survived", ascending = False)    
  

#%% Aykırı değer tespiti (Outlier Detection)


def AykiriDegerTespitEtme(df, ozellikler):
    tespit_satir = []
    
    for o in ozellikler:
        #1. çeyrek
        Q1 = np.percentile(df[o], 25)
        #3. çeyrek
        Q3 = np.percentile(df[o], 75)
        #IQR (Interquartile Range)
        IQR = Q3 - Q1
        #Aykırı değer adimi
        aykirilikAdimi = IQR * 1.5
        
        aykiriDegerler = df[(df[o] < Q1 - aykirilikAdimi) | (df[o] > Q3 + aykirilikAdimi)].index
        
        tespit_satir.extend(aykiriDegerler)
    
    tespit_satir = Counter(tespit_satir)
    
    tekrarliTespitSatir = list(i for i,v in tespit_satir.items() if v > 2)
    
    return tekrarliTespitSatir
        

#aykiri değerlerini görüntüleme

train_df.loc[AykiriDegerTespitEtme(train_df, ["Age", "SibSp", "Parch", "Fare"])]
    
df = train_df.copy()    

train_df = train_df.drop(AykiriDegerTespitEtme(train_df, ["Age", "SibSp", "Parch", "Fare"]), axis = 0).reset_index(drop = True)    

 
#%% Eksik veri işleme (Missing value handling)

dfConcat = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)

dfConcat.columns[dfConcat.isnull().any()]

dfConcat.isnull().sum()

dfConcat.boxplot(column="Fare", by = "Embarked")

dfConcat["Embarked"] = dfConcat["Embarked"].fillna("C")

dfConcat[dfConcat["Fare"].isnull()]

dfConcat["Fare"] = dfConcat["Fare"].fillna(np.mean(dfConcat[dfConcat["Pclass"] == 3]["Fare"]))

### yas degerini doldurma

dfConcat[dfConcat["Age"].isnull()]

sns.catplot(x = "Gender", y = "Age", data = dfConcat, kind = "box")
plt.show()

sns.heatmap(dfConcat[["Age", "SibSp", "Parch", "Pclass"]].corr(), annot= True)
plt.show()

indexNanAge = list(dfConcat["Age"][dfConcat["Age"].isnull()].index)   

for i in indexNanAge:
    #aynı sibSp, Parch, Pclass
    age_pred = dfConcat["Age"][((dfConcat["SibSp"] == dfConcat.iloc[i]["SibSp"]) & (dfConcat["Parch"] == dfConcat.iloc[i]["Parch"]) & (dfConcat["Pclass"] == dfConcat.iloc[i]["Pclass"]))].median()    
    #genel durum
    age_med = dfConcat["Age"].median()
    
    if not np.isnan(age_pred):
        dfConcat["Age"].iloc[i] = age_pred
    else:
        dfConcat["Age"].iloc[i] = age_med
        
dfConcat[dfConcat["Age"].isnull()]    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


