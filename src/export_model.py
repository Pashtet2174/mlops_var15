from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from project_utils import (
    DATASET_SOURCE,
    FEATURE_COLUMNS,
    MODEL_NAME,
    PREPROCESSING_VERSION,
    TARGET_COLUMN,
    load_dataset,
    sha256_file,
    write_split_file,
)


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    # Resolve the movable alias exactly once, then load the immutable version number.
    registry_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    immutable_uri = f"models:/{MODEL_NAME}/{registry_version.version}"
    model = mlflow.sklearn.load_model(immutable_uri)

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "exported_model"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(out_dir),
        pip_requirements=requirements,
    )

    frame, X, y = load_dataset()
    split = write_split_file(len(frame))
    test_indices = split["test_indices"]
    X_test = X.iloc[test_indices]
    sample_X = X_test.head(3)
    sample_records = [
        {column: float(value) for column, value in record.items()}
        for record in sample_X.to_dict(orient="records")
    ]
    expected = [float(value) for value in model.predict(sample_X).tolist()]

    run = client.get_run(registry_version.run_id)
    metadata = {
        "name": MODEL_NAME,
        "model_version": str(registry_version.version),
        "run_id": registry_version.run_id,
        "algorithm": run.data.params.get("algorithm"),
        "model_params": run.data.params,
        "target_variable": TARGET_COLUMN,
        "target_name": "Heating Load",
        "features": FEATURE_COLUMNS,
        "excluded_from_features": ["Y2"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_definition": run.data.params.get("preprocessing_definition"),
        "dataset_source": DATASET_SOURCE,
        "dataset_sha256": sha256_file(),
        "split_random_state": 42,
        "selection_metric": "val_mae",
        "validation_mae": run.data.metrics.get("val_mae"),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (root / "sample_request.json").write_text(
        json.dumps({"instances": sample_records}, indent=2), encoding="utf-8"
    )
    (root / "expected_predictions.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))
    print("Exported immutable model URI:", immutable_uri)


if __name__ == "__main__":
    main()
