import argparse
import mlflow
from mlflow.exceptions import MlflowException


def parse_args():
	parser = argparse.ArgumentParser(description="MLflow Model Registry işlemleri")
	parser.add_argument("--model-name", required=True, type=str, help="Kayıtlı model adı")
	parser.add_argument("--run-id", required=True, type=str, help="Kaydedilecek çalışmanın run_id değeri")
	parser.add_argument("--artifact-path", type=str, default="model", help="Çalışma içindeki model artifact yolu")
	parser.add_argument("--stage", type=str, default="Staging", choices=["None", "Staging", "Production", "Archived"], help="Geçiş yapılacak aşama")
	parser.add_argument("--archive-existing", action="store_true", help="Var olan sürümleri arşivle")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	model_uri = f"runs:/{args.run_id}/{args.artifact_path}"
	print(f"Registering model from URI: {model_uri}")

	try:
		result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
		version = result.version
		print(f"Registered model '{args.model_name}' version={version}")
	except MlflowException as e:
		print(f"Register failed: {e}")
		return

	# Stage'e geçiş
	if args.stage != "None":
		client = mlflow.tracking.MlflowClient()
		client.transition_model_version_stage(
			name=args.model_name,
			version=version,
			stage=args.stage,
			archive_existing_versions=args.archive_existing,
		)
		print(f"Transitioned '{args.model_name}' v{version} to stage={args.stage}")


if __name__ == "__main__":
	main()
