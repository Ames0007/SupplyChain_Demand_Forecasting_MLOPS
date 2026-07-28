"""
MLflow experiment configuration and shared utilities.

Experiment naming convention:
  demand-forecasting/{model_name}/{feature_version}/{timestamp}

Model Registry stages:
  None → Staging → Production → Archived
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import sys
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "https://dagshub.com/amitabh1609/SupplyChain_Demand_Forecasting_MLOPS.mlflow"
)
os.environ.setdefault("MLFLOW_TRACKING_USERNAME", os.getenv("DAGSHUB_USERNAME", "amitabh1609"))
os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", os.getenv("DAGSHUB_TOKEN", ""))
EXPERIMENT_BASE = "demand-forecasting"
REGISTERED_MODEL_NAME = "demand-forecasting-quantile-lgbm"


def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    return client


def get_or_create_experiment(model_name: str, feature_version: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    experiment_name = f"{EXPERIMENT_BASE}/{model_name}/{feature_version}/{timestamp}"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created MLflow experiment: %s", experiment_name)
    else:
        experiment_id = exp.experiment_id
        logger.info("Using existing MLflow experiment: %s", experiment_name)
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_environment() -> None:
    """Log Python + library versions for reproducibility."""
    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("platform", platform.platform())
    for lib in ["lightgbm", "xgboost", "mlflow", "pandas", "numpy", "scikit-learn"]:
        try:
            version = importlib.metadata.version(lib)
            mlflow.log_param(f"lib_{lib}", version)
        except importlib.metadata.PackageNotFoundError:
            pass


def log_standard_tags(
    model_type: str,
    feature_version: str,
    wape: float | None = None,
    promoted_by: str | None = None,
) -> None:
    mlflow.set_tag("model_type", model_type)
    mlflow.set_tag("feature_version", feature_version)
    if wape is not None:
        mlflow.set_tag("wape", f"{wape:.2f}")
    if promoted_by:
        mlflow.set_tag("promoted_by", promoted_by)
