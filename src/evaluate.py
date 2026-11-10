from __future__ import annotations

import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from project_utils import load_dataset, write_split_file


def main():
    root = Path(__file__).resolve().parents[1]
    model = mlflow.sklearn.load_model(str(root / "exported_model"))
    frame, X, y = load_dataset()
    split = write_split_file(len(frame))
    test_indices = split["test_indices"]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    predictions = model.predict(X_test)
    results = {
        "target_variable": "Y1",
        "n_test": len(test_indices),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "note": "Test set was evaluated after model selection; do not tune on this result.",
    }
    output_path = root / "test_metrics.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
