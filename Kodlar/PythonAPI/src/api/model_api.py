from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

app = FastAPI(title="MLflow Model Save API", version="1.0.0")


class TrainAndLogRequest(BaseModel):
	experiment_name: str
	model_name: Optional[str] = None  # for registry
	register: bool = False
	stage: Optional[str] = None  # e.g., "Staging" or "Production"
	test_size: float = 0.2
	random_state: int = 42


@app.post("/models/train-and-log")
async def train_and_log(body: TrainAndLogRequest):
	# Prepare data
	X, y = load_breast_cancer(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=body.test_size, random_state=body.random_state
	)

	# Set experiment and autolog
	mlflow.set_experiment(body.experiment_name)
	mlflow.sklearn.autolog(log_models=False)

	with mlflow.start_run() as run:
		model = LogisticRegression(max_iter=1000)
		model.fit(X_train, y_train)

		preds = model.predict(X_test)
		acc = accuracy_score(y_test, preds)
		mlflow.log_metric("accuracy", acc)

		# Build input example and signature
		input_example = pd.DataFrame(X_test[:2, :])
		signature = mlflow.models.signature.infer_signature(
			pd.DataFrame(X_train[:10, :]), pd.Series(model.predict(X_train[:10, :]))
		)

		# Log model artifact at "model"
		mlflow.sklearn.log_model(
			sk_model=model,
			artifact_path="model",
			input_example=input_example,
			signature=signature,
		)

		run_id = run.info.run_id
		model_uri = f"runs:/{run_id}/model"

		response = {
			"run_id": run_id,
			"model_uri": model_uri,
			"accuracy": acc,
		}

		if body.register:
			if not body.model_name:
				return {
					**response,
					"warning": "'register' true ancak 'model_name' verilmedi, kayıt atlandı.",
				}
			reg = mlflow.register_model(model_uri=model_uri, name=body.model_name)
			version = reg.version
			response["registered_model"] = body.model_name
			response["model_version"] = version

			if body.stage:
				client = mlflow.tracking.MlflowClient()
				client.transition_model_version_stage(
					name=body.model_name,
					version=version,
					stage=body.stage,
					archive_existing_versions=True,
				)
				response["stage"] = body.stage

		return response
