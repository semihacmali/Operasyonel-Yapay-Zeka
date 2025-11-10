# -*- coding: utf-8 -*-
"""

@author: SemihAcmali
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#  Adım 1: Dengesiz bir ikili sınıflandırma veri kümesi oluşturun
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=13)

np.unique(y, return_counts=True)

# Veri kümesini eğitim ve test kümelerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)


#Deneme 1: Lojistik Regresyon Sınıflandırıcısını Eğitin

# Model hiperparametrelerini tanımlayın
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 13,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_dict

import mlflow


mlflow.set_experiment("İlk Deneyim")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(lr, "Logistic Regression")  