"""
Model promotion gate: never blindly promote to Production.

Promotes a candidate model to Production only if it beats the current
Production model by at least `threshold` on the target metric.

Usage:
    python training/promote_model.py --run-id abc123 --metric wape --threshold 0.02
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
from mlops.model_registry import (
    get_production_run_id,
    get_production_wape,
    register_model,
    transition_stage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOWER_IS_BETTER = {"wape", "mae", "rmse", "mape", "smape"}
HIGHER_IS_BETTER = {"coverage_80pct"}


def get_metric_from_run(run_id: str, metric: str) -> float | None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = setup_mlflow()
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get(metric)
    except Exception as exc:
        logger.error("Could not fetch metric '%s' from run %s: %s", metric, run_id, exc)
        return None


def promote_model(
    candidate_run_id: str,
    metric: str = "wape",
    threshold: float = 0.02,
) -> dict:
    """
    Promote candidate model to Production only if it beats the current
    Production model by at least `threshold` on `metric`.

    If no Production model exists, promote automatically.
    If candidate is worse or ties within threshold, keep existing Production model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = setup_mlflow()

    candidate_value = get_metric_from_run(candidate_run_id, metric)
    if candidate_value is None:
        result = {
            "promotion_status": "rejected",
            "rejection_reason": f"Could not read metric '{metric}' from candidate run {candidate_run_id}",
            "candidate_run_id": candidate_run_id,
            "metric": metric,
        }
        _log_promotion_result(candidate_run_id, result)
        return result

    production_run_id = get_production_run_id()

    if production_run_id is None:
        # No production model — promote automatically
        version = register_model(candidate_run_id)
        transition_stage(version, "Production")
        result = {
            "promotion_status": "promoted",
            "reason": "No existing Production model — first promotion",
            "candidate_run_id": candidate_run_id,
            "candidate_version": version,
            f"candidate_{metric}": candidate_value,
            "metric": metric,
        }
        logger.info("PROMOTED: first model version %s to Production", version)
        _log_promotion_result(candidate_run_id, result, client)
        return result

    production_value = get_metric_from_run(production_run_id, metric)
    if production_value is None:
        # Production model has no recorded metric — promote anyway
        version = register_model(candidate_run_id)
        transition_stage(version, "Production")
        result = {
            "promotion_status": "promoted",
            "reason": "Production model had no recorded metric — promoting candidate",
            "candidate_run_id": candidate_run_id,
            "candidate_version": version,
            f"candidate_{metric}": candidate_value,
            "metric": metric,
        }
        _log_promotion_result(candidate_run_id, result, client)
        return result

    # Determine improvement
    if metric in LOWER_IS_BETTER:
        improvement = (production_value - candidate_value) / (production_value + 1e-8)
        is_better = improvement > threshold
    else:
        improvement = (candidate_value - production_value) / (production_value + 1e-8)
        is_better = improvement > threshold

    logger.info(
        "Candidate %s=%.4f vs Production %s=%.4f (improvement=%.2f%%, threshold=%.2f%%)",
        metric, candidate_value, metric, production_value,
        improvement * 100, threshold * 100,
    )

    if is_better:
        version = register_model(candidate_run_id)
        transition_stage(version, "Production")
        result = {
            "promotion_status": "promoted",
            "candidate_run_id": candidate_run_id,
            "candidate_version": version,
            f"candidate_{metric}": candidate_value,
            f"production_{metric}": production_value,
            "improvement_pct": round(improvement * 100, 4),
            "threshold_pct": round(threshold * 100, 4),
            "metric": metric,
        }
        logger.info("PROMOTED: version %s with %.2f%% improvement", version, improvement * 100)
    else:
        rejection_reason = (
            f"Candidate {metric}={candidate_value:.4f} does not improve "
            f"production {metric}={production_value:.4f} by threshold {threshold*100:.2f}%"
        )
        result = {
            "promotion_status": "rejected",
            "rejection_reason": rejection_reason,
            "candidate_run_id": candidate_run_id,
            f"candidate_{metric}": candidate_value,
            f"production_{metric}": production_value,
            "improvement_pct": round(improvement * 100, 4),
            "threshold_pct": round(threshold * 100, 4),
            "metric": metric,
        }
        logger.info("REJECTED: %s", rejection_reason)

    _log_promotion_result(candidate_run_id, result, client)
    return result


def _log_promotion_result(run_id: str, result: dict, client=None) -> None:
    """Log promotion outcome back to the candidate's MLflow run."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("promotion_status", result["promotion_status"])
            if "rejection_reason" in result:
                mlflow.set_tag("rejection_reason", result["rejection_reason"])
            mlflow.log_dict(result, "promotion_report.json")
    except Exception as exc:
        logger.warning("Could not log promotion result to MLflow: %s", exc)


def main():
    parser = argparse.ArgumentParser(description="Model promotion gate")
    parser.add_argument("--run-id", required=True, help="MLflow run ID of the candidate model")
    parser.add_argument("--metric", default="wape", help="Metric to compare (default: wape)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Minimum relative improvement required for promotion (default: 0.02 = 2%%)",
    )
    args = parser.parse_args()

    result = promote_model(
        candidate_run_id=args.run_id,
        metric=args.metric,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["promotion_status"] == "promoted" else 1)


if __name__ == "__main__":
    main()
