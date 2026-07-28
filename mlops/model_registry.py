"""
MLflow Model Registry operations: register, stage, promote, archive.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow.tracking import MlflowClient

from mlops.mlflow_config import REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def register_model(run_id: str, artifact_path: str = "model") -> str:
    """Register a model from a completed run into the Model Registry."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    logger.info("Registered model version %s from run %s", result.version, run_id)
    return result.version


def transition_stage(version: str, stage: str, archive_existing: bool = True) -> None:
    """
    Transition a model version to a new stage.
    Valid stages: Staging, Production, Archived, None
    """
    client = get_client()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    logger.info("Model version %s → stage: %s", version, stage)


def get_production_version() -> str | None:
    """Return the version number of the current Production model, or None."""
    client = get_client()
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        return versions[0].version if versions else None
    except Exception:
        return None


def get_production_run_id() -> str | None:
    """Return the MLflow run_id of the current Production model."""
    client = get_client()
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        return versions[0].run_id if versions else None
    except Exception:
        return None


def get_production_wape() -> float | None:
    """Fetch the WAPE tag from the Production model's run."""
    run_id = get_production_run_id()
    if run_id is None:
        return None
    client = get_client()
    try:
        run = client.get_run(run_id)
        wape_str = run.data.tags.get("wape")
        return float(wape_str) if wape_str else None
    except Exception:
        return None


def load_production_model():
    """Load the production model as a Python function using MLflow."""
    version = get_production_version()
    if version is None:
        raise RuntimeError("No Production model found in MLflow registry.")
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    logger.info("Loading production model version %s from %s", version, model_uri)
    return mlflow.lightgbm.load_model(model_uri)
