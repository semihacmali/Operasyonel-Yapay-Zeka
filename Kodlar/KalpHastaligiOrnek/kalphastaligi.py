# -*- coding: utf-8 -*-
"""
Kalp hastaligin teshisi icin random forest algoritmasinin kullanimi
"""

import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#random sayi uretim katsayisi
seed = 13

# veri setinin yüklenmesi
df = pd.read_csv("heart.csv")


# Sayisal olmayan özelliklerin tespiti

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
categorical_columns


# Sayisal ozelliklere donusturulmesi

le_model = LabelEncoder()

df['Sex'] = le_model.fit_transform(df['Sex'])
df['ChestPainType'] = le_model.fit_transform(df['ChestPainType'])
df['RestingECG'] = le_model.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le_model.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le_model.fit_transform(df['ST_Slope'])




# veri setinin train ve test olarak ayrilmasi
y = df.pop("HeartDisease")

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)


########################
######## MODEL #########
########################

model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=seed)
model.fit(X_train,y_train)

# Egitim dogruluk skoru
train_score = model.score(X_train, y_train) * 100

# Test dogruluk skoru
test_score = model.score(X_test, y_test) * 100

# Metrikleri dosyaya yazdirma
with open("metrikler.txt", 'w') as outfile:
        outfile.write("Egitim Dogruluk Degeri: %2.1f%%\n" % train_score)
        outfile.write("Test Dogruluk Degeri: %2.1f%%\n" % test_score)

###########################
##### Gorsellestirme ######
###########################

# Ozelliklerin oneminin hesaplanması

importances = model.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)


#grafigin olusturulmasi
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Önem',fontsize = axis_fs) 
ax.set_ylabel('Özellike', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nözellik önemi', fontsize = title_fs)

plt.tight_layout()
plt.savefig("ozellik_onem.png",dpi=120) 
plt.close()



