# modelling.py
import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train(data_path: Path):
    # 1) Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"]).values
    y = df["Outcome"].astype(int).values

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Set MLflow lokal (folder mlruns akan dibuat otomatis)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("diabetes-basic")

    # 4) Aktifkan autolog 
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=False)

    # 5) Train model
    with mlflow.start_run(run_name="logreg_baseline"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

    print("Training selesai. Cek folder ./mlruns atau buka MLflow UI di http://127.0.0.1:5000.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=str(
            Path(__file__).with_name("dataset_preprocessing") / "diabetes_preprocessed.csv"
        ),
        help="Path ke CSV hasil preprocessing",
    )
    args = parser.parse_args()
    train(Path(args.data))
