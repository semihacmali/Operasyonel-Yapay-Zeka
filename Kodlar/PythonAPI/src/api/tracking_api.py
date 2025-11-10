from fastapi import FastAPI
from pydantic import BaseModel
import mlflow

app = FastAPI(title="MLflow Tracking API", version="1.0.0")


class CreateRunRequest(BaseModel):
	experiment_name: str
	run_name: str | None = None


class LogParamRequest(BaseModel):
	run_id: str
	key: str
	value: str


class LogMetricRequest(BaseModel):
	run_id: str
	key: str
	value: float
	step: int | None = None


@app.post("/runs/create")
async def create_run(body: CreateRunRequest):
	exp = mlflow.set_experiment(body.experiment_name)
	with mlflow.start_run(run_name=body.run_name) as run:
		return {"run_id": run.info.run_id, "experiment_id": exp.experiment_id}


@app.post("/runs/log-param")
async def log_param(body: LogParamRequest):
	with mlflow.start_run(run_id=body.run_id):
		mlflow.log_param(body.key, body.value)
	return {"status": "ok"}


@app.post("/runs/log-metric")
async def log_metric(body: LogMetricRequest):
	with mlflow.start_run(run_id=body.run_id):
		mlflow.log_metric(body.key, body.value, step=body.step if body.step is not None else 0)
	return {"status": "ok"}
