import argparse
import os

import mlflow
from mlflow import MlflowClient

from project_utils import MODEL_NAME


def main():
    parser = argparse.ArgumentParser(description="Assign a Registry alias")
    parser.add_argument("--version", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, args.version)
    client.set_registered_model_alias(MODEL_NAME, args.alias, args.version)
    print(f"{args.alias} -> {MODEL_NAME} version {version.version}, run {version.run_id}")


if __name__ == "__main__":
    main()
