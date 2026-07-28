"""
Rollback to a previous registered model version.

Every rollback is logged as a named MLflow run for audit trail.

Usage:
    python mlops/rollback_model.py --to-version 5 --reason "WAPE regression on intermittent SKUs"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.mlflow_config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, setup_mlflow
from mlops.model_registry import get_production_version, transition_stage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def rollback(to_version: str, reason: str) -> dict:
    """
    Demote current Production model to Archived.
    Promote specified version back to Production.
    Log rollback event with reason and timestamp.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = setup_mlflow()

    current_production_version = get_production_version()
    if current_production_version is None:
        logger.warning("No current Production model found. Proceeding with rollback anyway.")

    # Archive current production
    if current_production_version and current_production_version != str(to_version):
        transition_stage(current_production_version, "Archived", archive_existing=False)
        logger.info("Archived current production version: %s", current_production_version)

    # Promote target version to Production
    transition_stage(str(to_version), "Production", archive_existing=True)
    logger.info("Rolled back to version %s", to_version)

    rollback_event = {
        "event_type": "rollback",
        "timestamp": datetime.utcnow().isoformat(),
        "rolled_back_from_version": current_production_version,
        "rolled_back_to_version": str(to_version),
        "reason": reason,
    }

    # Log rollback event as a dedicated MLflow run
    mlflow.set_experiment(f"demand-forecasting/rollback-events")
    with mlflow.start_run(run_name=f"rollback-to-v{to_version}") as run:
        mlflow.set_tag("event_type", "rollback")
        mlflow.set_tag("rollback_to_version", str(to_version))
        mlflow.set_tag("rollback_from_version", str(current_production_version))
        mlflow.set_tag("reason", reason)
        mlflow.log_dict(rollback_event, "rollback_event.json")
        logger.info("Logged rollback event to MLflow run %s", run.info.run_id)

    print(json.dumps(rollback_event, indent=2))
    return rollback_event


def main():
    parser = argparse.ArgumentParser(description="Roll back to a previous model version")
    parser.add_argument("--to-version", required=True, help="Target model version to restore")
    parser.add_argument("--reason", required=True, help="Reason for rollback (logged to audit trail)")
    args = parser.parse_args()
    rollback(to_version=args.to_version, reason=args.reason)


if __name__ == "__main__":
    main()
