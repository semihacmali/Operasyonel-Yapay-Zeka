# MLflow API Örnekleri

Bu depo, MLflow ile model deney takibi, model kaydı (Model Registry) ve servis etme senaryolarını hızla denemeniz için örnek Python komutları ve FastAPI servisleri içerir.

## Kurulum

1) Sanal ortam (önerilir):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Bağımlılıklar:
```powershell
pip install -r requirements.txt
```

3) MLflow izleme URI (opsiyonel): Yerel dosya tabanlı izleme için varsayılanı kullanabilirsiniz veya bir sunucuya yönlendirebilirsiniz.
```powershell
# Örn. dosya tabanlı (varsayılan): mlruns klasörü oluşturulur
# setx MLFLOW_TRACKING_URI "file://${PWD}/mlruns"

# Örn. uzak MLflow sunucusu:
# setx MLFLOW_TRACKING_URI "http://localhost:5000"
```

## Örnekler

- Temel deney takibi: `src/tracking_basic.py`
- scikit-learn ile autolog: `src/training_sklearn_autolog.py`
- Model Registry işlemleri: `src/registry_ops.py`
- FastAPI ile MLflow takip servisi: `src/api/tracking_api.py`
- FastAPI ile model servis etme: `src/api/predict_api.py`

### 1) Temel Deney Takibi
```powershell
python src/tracking_basic.py --experiment-name "DemoExp" --alpha 0.1 --n-epochs 5
```

### 2) scikit-learn Autolog Örneği
```powershell
python src/training_sklearn_autolog.py --experiment-name "SKLearnDemo" --test-size 0.2 --random-state 42
```

### 3) Model Registry İşlemleri
```powershell
# Bir çalışmadaki modeli kayıtlı modele dönüştürme ve aşama geçişleri
python src/registry_ops.py ^
  --model-name "credit-risk-model" ^
  --run-id "<MEVCUT_RUN_ID>" ^
  --stage "Staging"
```

`<MEVCUT_RUN_ID>` değerini MLflow UI'dan veya konsol çıktısından alabilirsiniz.

### 4) FastAPI Takip Servisi (parametre, metrik loglama)
```powershell
uvicorn src.api.tracking_api:app --host 0.0.0.0 --port 8000 --reload
```
İstek örnekleri `examples/http/tracking_api.ps1` dosyasında mevcuttur.

### 5) FastAPI Tahmin Servisi (MLflow modeli yükleyerek)
```powershell
# Model URI örnekleri:
# "runs:/<RUN_ID>/model"
# "models:/credit-risk-model/Production"

uvicorn src.api.predict_api:app --host 0.0.0.0 --port 8001 --reload
```
İstek örnekleri `examples/http/predict_api.ps1` dosyasında mevcuttur.

## MLflow UI

Yerel dosya tabanlı kullanımda UI'ı hızlıca açmak için:
```powershell
mlflow ui --backend-store-uri "file://${PWD}/mlruns" --host 0.0.0.0 --port 5000
```

## Notlar
- PowerShell kullanan Windows ortamları için örnek komutlar `.ps1` dosyaları ile verilmiştir.
- İzleme URI boş bırakılırsa MLflow, çalışma klasörünüzde `mlruns/` oluşturur.
