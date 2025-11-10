# -*- coding: utf-8 -*-
"""


@author: SemihAcmali
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


#  Adım 1: Dengesiz bir ikili sınıflandırma veri kümesi oluşturun
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=13)

np.unique(y, return_counts=True)

# Veri kümesini eğitim ve test kümelerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=13)


#Deney 1: Lojistik Regresyon Sınıflandırıcısını Eğitin

log_reg = LogisticRegression(C=1, solver='liblinear')
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print(classification_report(y_test, y_pred_log_reg))

#Deney 2: Rastgele Orman Sınıflandırıcısını Eğitin
rf_clf = RandomForestClassifier(n_estimators=30, max_depth=3)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

#Deney 3: XGBoost'u Eğitin

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb))

#Deney 4: SMOTETomek kullanarak sınıf dengesizliğini ele alın ve ardından XGBoost'u eğitin

from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

np.unique(y_train_res, return_counts=True)

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb))


#%% MLFlow Kullanarak Deneyleri İzleme

models = [
    (
        "Logistic Regression", 
        LogisticRegression(C=1, solver='liblinear'), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest", 
        RandomForestClassifier(n_estimators=30, max_depth=3), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier With SMOTE",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'), 
        (X_train_res, y_train_res),
        (X_test, y_test)
    )
]

reports = []

for model_name, model, train_set, test_set in models:
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)
    
    


import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import re

def model_name_bosluk_silme(name):
    return re.sub(r"[\s\-.]", '-', name)


# Initialize MLflow
tracking_uri = "http://192.168.1.107:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Anomali Tespiti")

#MlFlow Client API
client = MlflowClient(tracking_uri=tracking_uri)

run_ids = {}

for i, element in enumerate(models):
    model_name = element[0]
    model = element[1]
    report = reports[i]
    
    with mlflow.start_run(run_name=model_name) as run:        
        mlflow.log_param("model", model_name)
        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('recall_class_1', report['1']['recall'])
        mlflow.log_metric('recall_class_0', report['0']['recall'])
        mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])
        signature = infer_signature(X_train, model.predict(X_train))        
        
        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
        
        #run id yi yakalama    
        run_ids[model_name] = run.info.run_id
        run_id = run.info.run_id
        registered_model_name = model_name_bosluk_silme(model_name)
        source_arficath_path = "model"
        source_model_uri = f"runs:/{run_id}/{source_arficath_path}"
        
        try:
            client.get_registered_model(registered_model_name)
        except Exception:
            client.create_registered_model(registered_model_name)
            
        model_version = client.create_model_version(
            name = registered_model_name,
            source = source_model_uri,
            run_id = run_id)
        
        print(f"Model versiyonu kaydedildi: {registered_model_name} v{model_version.version}")
        



target_model_name = model_name_bosluk_silme("XGBClassifier-With-SMOTE")

latest_versions = client.get_latest_versions(target_model_name, stages= ["None"])

if latest_versions:
    latest_version = latest_versions[0]
    model_uri_load = f"models:/{target_model_name}/{latest_version.version}"
    loaded_model = mlflow.xgboost.load_model(model_uri_load)
    y_predict = loaded_model.predict(X_test)
    print(f"\n {target_model_name} v{latest_version.version} yüklendi.")
    print(f"\n ilk 10 tahmin : {y_predict[:10]}")
else:
    print(f"Model versiyonu bulunamadı: {target_model_name}")

# #Modelleri kaydetme

# model_name = "XGB-Smote"
# run_id = input("Run Id Giriniz :")
# source_arficath_path = "model"

# model_uri = f"runs:/{run_id}/{source_arficath_path}"
# result = mlflow.register_model(model_uri, model_name)


































