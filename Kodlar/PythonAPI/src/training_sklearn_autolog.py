import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def parse_args():
	parser = argparse.ArgumentParser(description="scikit-learn + MLflow autolog örneği")
	parser.add_argument("--experiment-name", type=str, default="SKLearnDemo")
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--random-state", type=int, default=42)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	mlflow.set_experiment(args.experiment_name)
	mlflow.sklearn.autolog(log_models=True)

	X, y = load_breast_cancer(return_X_y=True)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=args.random_state
	)

	with mlflow.start_run() as run:
		model = LogisticRegression(max_iter=1000, n_jobs=None)
		model.fit(X_train, y_train)

		preds = model.predict(X_test)
		acc = accuracy_score(y_test, preds)
		mlflow.log_metric("accuracy", acc)
		print(f"Run: {run.info.run_id} accuracy={acc:.4f}")


if __name__ == "__main__":
	main()
