from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ENB2012_data.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_PATH = PROCESSED_DIR / "split.json"

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
NUMERIC_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X7"]
CATEGORICAL_COLUMNS = ["X6", "X8"]
TARGET_COLUMN = "Y1"
EXCLUDED_TARGET_COLUMN = "Y2"
MODEL_NAME = "energy-heating-regressor"
EXPERIMENT_NAME = "lab4-energy-efficiency-y1"
PREPROCESSING_VERSION = "energy-preprocessing-v1"
DATASET_SOURCE = "https://archive.ics.uci.edu/dataset/242/energy+efficiency"
RANDOM_STATE = 42


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read the original workbook. Y2 is deliberately never returned in X."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Put ENB2012_data.xlsx in data/raw/."
        )

    frame = pd.read_excel(DATA_PATH, sheet_name=0, engine="openpyxl")
    expected = FEATURE_COLUMNS + [TARGET_COLUMN, EXCLUDED_TARGET_COLUMN]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"The workbook is missing expected columns: {missing}")

    # Select only X1-X8. Y2 is not part of model input or training labels.
    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET_COLUMN].astype(float).copy()

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain NaN or infinite values.")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("Target Y1 contains NaN or infinite values.")

    return frame, X, y


def sha256_file(path: Path = DATA_PATH) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_split(n_rows: int) -> dict[str, list[int]]:
    """Reproduce the manual's 60/20/20 split using one fixed random seed."""
    indices = np.arange(n_rows)
    development_indices, test_indices = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    train_indices, validation_indices = train_test_split(
        development_indices, test_size=0.25, random_state=RANDOM_STATE, shuffle=True
    )
    return {
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }


def split_metadata(n_rows: int) -> dict[str, Any]:
    split = make_split(n_rows)
    return {
        **split,
        "dataset_rows": n_rows,
        "random_state": RANDOM_STATE,
        "shuffle": True,
        "proportions": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "excluded_from_features": [EXCLUDED_TARGET_COLUMN],
    }


def write_split_file(n_rows: int) -> dict[str, Any]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = split_metadata(n_rows)
    SPLIT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_model(
    algorithm: str,
    alpha: float = 1.0,
    trees: int = 200,
    depth: int = 10,
):
    """Create one sklearn Pipeline with all preprocessing inside it."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    preprocessing_description = (
        "numeric:median_imputer+standard_scaler;"
        "categorical:most_frequent_imputer+onehot(handle_unknown=ignore)"
    )

    if algorithm == "ridge":
        numeric_steps.append(("scaler", StandardScaler()))
        estimator = Ridge(alpha=alpha)
    elif algorithm == "rf":
        estimator = RandomForestRegressor(
            n_estimators=trees,
            max_depth=depth,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        preprocessing_description = (
            "numeric:median_imputer;"
            "categorical:most_frequent_imputer+onehot(handle_unknown=ignore)"
        )
    else:
        raise ValueError("algorithm must be 'ridge' or 'rf'")

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        [("preprocessing", preprocessor), ("regressor", estimator)]
    )
    return pipeline, preprocessing_description
