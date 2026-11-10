from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/exported_model"))
model = mlflow.sklearn.load_model(str(MODEL_DIR))
metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
feature_names = metadata["features"]
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Energy Efficiency - Heating Load API",
    version="1.0.0",
    description="Predicts heating load Y1 from building geometry features X1-X8.",
)


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"]}


@app.get("/metadata")
def get_metadata():
    return metadata


@app.post("/predict")
def predict(body: PredictRequest):
    started = time.perf_counter()
    required = set(feature_names)

    for row_number, row in enumerate(body.instances):
        actual = set(row)
        if actual != required:
            missing = sorted(required - actual)
            extra = sorted(actual - required)
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Each instance must contain exactly X1-X8",
                    "row": row_number,
                    "missing_features": missing,
                    "extra_features": extra,
                    "required_features": feature_names,
                },
            )
        if not np.isfinite(np.asarray(list(row.values()), dtype=float)).all():
            raise HTTPException(status_code=422, detail="All feature values must be finite")

    X = pd.DataFrame(body.instances, columns=feature_names)
    predictions = [float(value) for value in model.predict(X).tolist()]
    latency_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "prediction model_version=%s rows=%s latency_ms=%.3f",
        metadata["model_version"],
        len(X),
        latency_ms,
    )
    return {
        "predictions": predictions,
        "target_variable": metadata["target_variable"],
        "model_version": metadata["model_version"],
        "latency_ms": round(latency_ms, 3),
    }
