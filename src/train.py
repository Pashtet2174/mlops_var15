from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error

from project_utils import (
    DATA_PATH,
    DATASET_SOURCE,
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    MODEL_NAME,
    PREPROCESSING_VERSION,
    TARGET_COLUMN,
    build_model,
    load_dataset,
    sha256_file,
    write_split_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train one Energy Efficiency regressor")
    parser.add_argument("--algorithm", choices=["ridge", "rf"], required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    frame, X, y = load_dataset()
    split = write_split_file(len(frame))
    train_idx = split["train_indices"]
    val_idx = split["validation_indices"]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model, preprocessing_description = build_model(
        args.algorithm, alpha=args.alpha, trees=args.trees, depth=args.depth
    )
    dataset_hash = sha256_file(DATA_PATH)
    run_name = (
        f"ridge-alpha-{args.alpha:g}"
        if args.algorithm == "ridge"
        else f"rf-trees-{args.trees}-depth-{args.depth}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "algorithm": args.algorithm,
            "random_state": 42,
            "dataset_rows": len(frame),
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "n_test": len(split["test_indices"]),
            "target_variable": TARGET_COLUMN,
            "excluded_target": "Y2",
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_definition": preprocessing_description,
        }
        if args.algorithm == "ridge":
            params["alpha"] = args.alpha
        else:
            params.update({"n_estimators": args.trees, "max_depth": args.depth})
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "dataset": "UCI Energy Efficiency",
                "dataset_source": DATASET_SOURCE,
                "dataset_sha256": dataset_hash,
                "purpose": "lab4-variant-15",
                "target_variable": TARGET_COLUMN,
                "excluded_target": "Y2",
                "preprocessing_version": PREPROCESSING_VERSION,
                "selection_metric": "val_mae",
            }
        )

        # Fit preprocessing and model only on the training portion.
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_predictions)))
        mlflow.log_metrics({"val_mae": float(val_mae), "val_rmse": val_rmse})

        # The MLflow input contains X1-X8 and Y1 only. Y2 is never passed as X.
        data_for_mlflow = X.copy()
        data_for_mlflow[TARGET_COLUMN] = y
        mlflow.log_input(
            mlflow.data.from_pandas(
                data_for_mlflow,
                source=DATASET_SOURCE,
                targets=TARGET_COLUMN,
            ),
            context="training_dataset",
        )
        mlflow.log_dict(split, "split_indices.json")
        mlflow.log_artifact(str(DATA_PATH), artifact_path="data_source")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")
        mlflow.log_artifact(str(Path(__file__).with_name("project_utils.py").resolve()), artifact_path="source")
        mlflow.log_artifact(str(Path(__file__).resolve().parents[1] / "requirements.txt"), artifact_path="source")

        validation_results = pd.DataFrame(
            {
                "actual_Y1": y_val.to_numpy(),
                "predicted_Y1": val_predictions,
                "absolute_error": np.abs(y_val.to_numpy() - val_predictions),
            }
        )
        validation_path = Path("validation_predictions.csv")
        validation_results.to_csv(validation_path, index=False)
        mlflow.log_artifact(str(validation_path))
        validation_path.unlink(missing_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_val, val_predictions, alpha=0.7)
        lower = float(min(y_val.min(), np.min(val_predictions)))
        upper = float(max(y_val.max(), np.max(val_predictions)))
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlabel("Actual Y1 (heating load)")
        ax.set_ylabel("Predicted Y1")
        ax.set_title(f"Validation: {args.algorithm}; MAE={val_mae:.4f}")
        fig.tight_layout()
        mlflow.log_figure(fig, "validation_actual_vs_predicted.png")
        plt.close(fig)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(3),
            signature=infer_signature(X_train, model.predict(X_train)),
            pip_requirements=(
                Path(__file__).resolve().parents[1] / "requirements.txt"
            ).read_text(encoding="utf-8").splitlines(),
        )

        output = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "algorithm": args.algorithm,
            "val_mae": float(val_mae),
            "val_rmse": val_rmse,
        }

        if args.register:
            registered = mlflow.register_model(model_info.model_uri, MODEL_NAME)
            client = mlflow.tracking.MlflowClient()
            version = str(registered.version)
            for key, value in {
                "target_variable": TARGET_COLUMN,
                "preprocessing_version": PREPROCESSING_VERSION,
                "dataset_sha256": dataset_hash,
                "algorithm": args.algorithm,
                "validation_mae": f"{val_mae:.10f}",
            }.items():
                client.set_model_version_tag(MODEL_NAME, version, key, value)
            output["registered_model"] = MODEL_NAME
            output["registered_version"] = version

        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
