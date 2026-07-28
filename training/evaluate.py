"""
Evaluation metrics for the demand forecasting leaderboard.

Implements WAPE, RMSE, MAE, MAPE, sMAPE, Bias, and quantile coverage metrics.
Also computes per-segment metrics (fast_moving, slow_moving, intermittent, seasonal)
to surface where models win and lose — intermittent SKUs are almost always hardest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

EPSILON = 1e-8  # Avoid division by zero in percentage metrics


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > EPSILON
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom)) * 100


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = np.sum(np.abs(y_true)) + EPSILON
    return float(np.sum(np.abs(y_true - y_pred)) / total) * 100


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_p10: np.ndarray | None = None,
    y_p90: np.ndarray | None = None,
) -> dict:
    """
    Compute all metrics for a single forecast array.

    Returns
    -------
    dict with keys: mae, rmse, mape, smape, wape, bias, coverage_80pct, interval_width
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Drop rows where either array is NaN
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) == 0:
        return {k: np.nan for k in ["mae", "rmse", "mape", "smape", "wape", "bias", "coverage_80pct", "interval_width"]}

    results = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _safe_mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "wape": _wape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),  # positive = over-forecast
    }

    if y_p10 is not None and y_p90 is not None:
        y_p10 = np.asarray(y_p10, dtype=float)[valid]
        y_p90 = np.asarray(y_p90, dtype=float)[valid]
        in_interval = (y_true >= y_p10) & (y_true <= y_p90)
        results["coverage_80pct"] = float(in_interval.mean()) * 100
        results["interval_width"] = float(np.mean(y_p90 - y_p10))
    else:
        results["coverage_80pct"] = np.nan
        results["interval_width"] = np.nan

    return results


def compute_segment_metrics(
    df: pd.DataFrame,
    pred_col: str = "p50",
    p10_col: str | None = None,
    p90_col: str | None = None,
) -> dict[str, dict]:
    """
    Compute metrics separately for each SKU demand category.

    df must have columns: [demand (actual), category, <pred_col>]
    """
    results = {}
    for category, grp in df.groupby("category"):
        y_true = grp["demand"].values
        y_pred = grp[pred_col].values
        y_p10 = grp[p10_col].values if p10_col and p10_col in grp else None
        y_p90 = grp[p90_col].values if p90_col and p90_col in grp else None
        results[str(category)] = compute_metrics(y_true, y_pred, y_p10, y_p90)
    return results


def build_leaderboard(
    results: list[dict],
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build and optionally save the model comparison leaderboard table.

    Parameters
    ----------
    results : list of dicts, each with keys:
        model, wape, rmse, bias, coverage_80pct, inference_ms
    """
    df = pd.DataFrame(results)
    df = df.sort_values("wape")

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info("Leaderboard saved → %s", save_path)

    return df


def log_segment_metrics_to_mlflow(segment_metrics: dict, run) -> None:
    """Log per-segment metrics as an MLflow JSON artifact."""
    try:
        import mlflow
        run.log_dict(segment_metrics, "segment_metrics.json")
        for segment, metrics in segment_metrics.items():
            for metric_name, value in metrics.items():
                if not np.isnan(value):
                    mlflow.log_metric(f"{segment}_{metric_name}", value)
    except Exception as exc:
        logger.warning("Could not log segment metrics to MLflow: %s", exc)
