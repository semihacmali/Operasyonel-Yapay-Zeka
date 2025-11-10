import argparse
import os
import time
import mlflow


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Temel MLflow takip örneği")
	parser.add_argument("--experiment-name", type=str, default="DemoExperiment")
	parser.add_argument("--alpha", type=float, default=0.1)
	parser.add_argument("--n-epochs", type=int, default=5)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# Deney oluştur/aoç
	experiment_id = mlflow.set_experiment(args.experiment_name).experiment_id
	print(f"Experiment set: {args.experiment_name} (id={experiment_id})")

	with mlflow.start_run() as run:
		print(f"Started run: {run.info.run_id}")

		# Parametreleri logla
		mlflow.log_param("alpha", args.alpha)
		mlflow.log_param("n_epochs", args.n_epochs)

		loss = 1.0
		for epoch in range(1, args.n_epochs + 1):
			# sahte eğitim döngüsü
			time.sleep(0.2)
			loss = loss * (1 - args.alpha)
			mlflow.log_metric("loss", loss, step=epoch)
			print(f"epoch={epoch}\tloss={loss:.4f}")

		# küçük bir artifact oluştur
		artifact_dir = "artifacts"
		os.makedirs(artifact_dir, exist_ok=True)
		artifact_path = os.path.join(artifact_dir, "notes.txt")
		with open(artifact_path, "w", encoding="utf-8") as f:
			f.write("Bu, temel bir MLflow artifact dosyasıdır.\n")
		mlflow.log_artifact(artifact_path, artifact_path="training_notes")

		print("Run completed.")


if __name__ == "__main__":
	main()
