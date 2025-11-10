import os
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

MODEL_URI_ENV = "MODEL_URI"

app = FastAPI(title="MLflow Predict API", version="1.0.0")
model = None


class PredictRequest(BaseModel):
	# JSON list of records or dict-of-lists compatible with pandas.DataFrame
	data: Any


@app.on_event("startup")
async def load_model_on_startup():
	global model
	model_uri = os.getenv(MODEL_URI_ENV)
	if not model_uri:
		raise RuntimeError(f"{MODEL_URI_ENV} environment variable is not set")
	model = mlflow.pyfunc.load_model(model_uri)


@app.post("/predict")
async def predict(body: PredictRequest):
	if model is None:
		raise HTTPException(status_code=500, detail="Model not loaded")
	try:
		df = pd.DataFrame(body.data)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Invalid data format: {e}")
	preds = model.predict(df)
	# Convert numpy/pandas types to native Python
	return {"predictions": preds.tolist() if hasattr(preds, "tolist") else preds}
